#!/usr/bin/env python3
"""Convert legged_lab T800 get-up pickles to the tracking NPZ schema.

The source files are joblib pickles produced by legged_lab's Isaac Sim
converter.  They contain root pose, 25 T800 joint positions, and six key-body
positions.  The tracking task needs the complete canonical 34-body state plus
velocities, so this script rebuilds those fields from the repository's T800
URDF and retimes every clip to a uniform output rate.

This conversion is CPU-only; Isaac Sim is not required.  The input directory
is read-only and the output is written atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from validate_t800_motion_fk import Joint, forward_kinematics, load_urdf_tree
except ModuleNotFoundError:  # Allow ``python -m scripts.convert_t800_getup_to_npz``.
    from scripts.validate_t800_motion_fk import Joint, forward_kinematics, load_urdf_tree


DEFAULT_SOURCE_DIR = Path(
    "/srv/shared/home/xjh/ws_kyx/legged_lab/source/legged_lab/legged_lab/data/MotionData/t800/amp/get_up"
)
DEFAULT_OUTPUT_DIR = Path("data/npz/t800_get_up")
DEFAULT_URDF = Path("source/whole_body_tracking/whole_body_tracking/assets/t800/urdf/serial_t800.urdf")
DEFAULT_OUTPUT_FPS = 50.0

CANONICAL_BODY_NAMES = (
    "LINK_BASE",
    "LINK_HIP_PITCH_L",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_YAW_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_L_TOE",
    "LINK_ANKLE_ROLL_L_HEEL",
    "LINK_HIP_PITCH_R",
    "LINK_HIP_ROLL_R",
    "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_ANKLE_ROLL_R_TOE",
    "LINK_ANKLE_ROLL_R_HEEL",
    "LINK_TORSO_YAW",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    "LINK_WRIST_PITCH_L",
    "LINK_WRIST_ROLL_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_SHOULDER_ROLL_R",
    "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_WRIST_PITCH_R",
    "LINK_WRIST_ROLL_R",
    "LINK_HEAD_PITCH",
    "LINK_HEAD_YAW",
)

BODY_ALIASES = {
    "LINK_ANKLE_ROLL_L_TOE": "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_L_HEEL": "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R_TOE": "LINK_ANKLE_ROLL_R",
    "LINK_ANKLE_ROLL_R_HEEL": "LINK_ANKLE_ROLL_R",
    "LINK_WRIST_PITCH_L": "LINK_ELBOW_YAW_L",
    "LINK_WRIST_ROLL_L": "LINK_ELBOW_YAW_L",
    "LINK_WRIST_PITCH_R": "LINK_ELBOW_YAW_R",
    "LINK_WRIST_ROLL_R": "LINK_ELBOW_YAW_R",
}


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("root_rot contains a zero-length quaternion")
    result /= norms
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] = -result[index]
    return result


def load_source_motion(path: Path) -> dict[str, Any]:
    """Load and validate one legged_lab joblib motion file."""

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - depends on the caller's env.
        raise RuntimeError("joblib is required to read the legged_lab T800 .pkl files") from exc

    data = joblib.load(path)
    required = {"fps", "root_pos", "root_rot", "dof_pos", "key_body_pos", "key_body_names"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"{path}: missing source fields {sorted(missing)}")

    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_quat = np.asarray(data["root_rot"], dtype=np.float64)
    joint_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    key_body_pos = np.asarray(data["key_body_pos"], dtype=np.float64)
    key_body_names = tuple(str(name) for name in data["key_body_names"])

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{path}: root_pos must have shape [frames, 3], got {root_pos.shape}")
    if root_quat.shape != (root_pos.shape[0], 4):
        raise ValueError(f"{path}: root_rot must have shape [{root_pos.shape[0]}, 4], got {root_quat.shape}")
    if joint_pos.shape != (root_pos.shape[0], 25):
        raise ValueError(f"{path}: dof_pos must have shape [{root_pos.shape[0]}, 25], got {joint_pos.shape}")
    if key_body_pos.shape != (root_pos.shape[0], len(key_body_names), 3):
        raise ValueError(f"{path}: key_body_pos shape does not match key_body_names: {key_body_pos.shape}")
    if root_pos.shape[0] < 2:
        raise ValueError(f"{path}: at least two frames are required")

    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{path}: fps must be positive, got {fps}")
    for name, array in (
        ("root_pos", root_pos),
        ("root_rot", root_quat),
        ("dof_pos", joint_pos),
        ("key_body_pos", key_body_pos),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: {name} contains non-finite values")

    return {
        "fps": fps,
        "root_pos": root_pos,
        "root_quat": _normalize_quaternions(root_quat),
        "joint_pos": joint_pos,
        "key_body_pos": key_body_pos,
        "key_body_names": key_body_names,
        "loop_mode": int(np.asarray(data.get("loop_mode", 0)).reshape(-1)[0]),
    }


def _build_body_kinematics(
    root_link: str,
    joints: list[Joint],
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
    body_names: tuple[str, ...] = CANONICAL_BODY_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.empty((len(joint_pos), len(body_names), 3), dtype=np.float64)
    quaternions = np.empty((len(joint_pos), len(body_names), 4), dtype=np.float64)

    for frame in range(len(joint_pos)):
        transforms = forward_kinematics(root_link, joints, root_pos[frame], root_quat[frame], joint_pos[frame])
        for body_index, body_name in enumerate(body_names):
            source_name = BODY_ALIASES.get(body_name, body_name)
            if source_name not in transforms:
                raise ValueError(f"T800 URDF does not contain body required by tracking schema: {source_name}")
            position, rotation = transforms[source_name]
            positions[frame, body_index] = position
            xyzw = Rotation.from_matrix(rotation).as_quat()
            quaternion = xyzw[[3, 0, 1, 2]]
            if frame > 0 and np.dot(quaternion, quaternions[frame - 1, body_index]) < 0.0:
                quaternion = -quaternion
            quaternions[frame, body_index] = quaternion

    return positions, quaternions


def _linear_interpolate(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flat_values = values.reshape(len(values), -1)
    clipped_times = np.clip(target_times, source_times[0], source_times[-1])
    upper = np.searchsorted(source_times, clipped_times, side="right")
    upper = np.clip(upper, 1, len(source_times) - 1)
    lower = upper - 1
    alpha = (clipped_times - source_times[lower]) / (source_times[upper] - source_times[lower])
    result = flat_values[lower] * (1.0 - alpha[:, None]) + flat_values[upper] * alpha[:, None]
    return result.reshape((len(target_times),) + values.shape[1:])


def _slerp_interpolate(quaternions: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = _normalize_quaternions(quaternions)
    clipped_times = np.clip(target_times, source_times[0], source_times[-1])
    upper = np.searchsorted(source_times, clipped_times, side="right")
    upper = np.clip(upper, 1, len(source_times) - 1)
    lower = upper - 1
    alpha = (clipped_times - source_times[lower]) / (source_times[upper] - source_times[lower])

    q0 = values[lower]
    q1 = values[upper].copy()
    dot = np.sum(q0 * q1, axis=-1)
    q1[dot < 0.0] *= -1.0
    dot = np.clip(np.sum(q0 * q1, axis=-1), -1.0, 1.0)

    lerp = q0 * (1.0 - alpha[:, None]) + q1 * alpha[:, None]
    lerp /= np.maximum(np.linalg.norm(lerp, axis=-1, keepdims=True), 1.0e-8)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    safe_sin_theta = np.where(np.abs(sin_theta) < 1.0e-8, 1.0, sin_theta)
    factor_0 = np.sin((1.0 - alpha) * theta) / safe_sin_theta
    factor_1 = np.sin(alpha * theta) / safe_sin_theta
    slerp = factor_0[:, None] * q0 + factor_1[:, None] * q1
    slerp /= np.maximum(np.linalg.norm(slerp, axis=-1, keepdims=True), 1.0e-8)
    return np.where((np.abs(dot) > 0.9995)[:, None], lerp, slerp)


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _angular_velocity(quaternions: np.ndarray, dt: float) -> np.ndarray:
    if len(quaternions) < 2:
        raise ValueError("At least two quaternion frames are required to compute angular velocity")
    interval = np.empty((len(quaternions) - 1, quaternions.shape[1], 3), dtype=np.float64)
    for frame in range(len(quaternions) - 1):
        for body in range(quaternions.shape[1]):
            q0 = quaternions[frame, body]
            q1 = quaternions[frame + 1, body]
            if np.dot(q0, q1) < 0.0:
                q1 = -q1
            delta = _quaternion_multiply(q1, np.asarray([q0[0], -q0[1], -q0[2], -q0[3]]))
            delta /= np.linalg.norm(delta)
            vector_norm = np.linalg.norm(delta[1:])
            if vector_norm < 1.0e-12:
                interval[frame, body] = 0.0
            else:
                angle = 2.0 * np.arctan2(vector_norm, max(delta[0], 0.0))
                interval[frame, body] = delta[1:] / vector_norm * angle / dt

    output = np.empty((len(quaternions), quaternions.shape[1], 3), dtype=np.float64)
    output[0] = interval[0]
    output[-1] = interval[-1]
    output[1:-1] = 0.5 * (interval[:-1] + interval[1:])
    return output


def _output_times(frame_count: int, source_fps: float, output_fps: float) -> np.ndarray:
    source_duration = (frame_count - 1) / source_fps
    scaled_duration = source_duration * output_fps
    rounded = round(scaled_duration)
    intervals = rounded if abs(scaled_duration - rounded) < 1.0e-6 else int(np.ceil(scaled_duration))
    intervals = max(intervals, 1)
    # If the source duration is not an exact multiple of the target period,
    # the final target sample is a short terminal hold at the source endpoint.
    return np.minimum(np.arange(intervals + 1, dtype=np.float64) / output_fps, source_duration)


def convert_motion(
    source: Path,
    output: Path,
    root_link: str,
    joints: list[Joint],
    output_fps: float,
    overwrite: bool,
) -> tuple[int, float, float]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists; pass --overwrite to replace it: {output}")

    source_motion = load_source_motion(source)
    source_root_pos = source_motion["root_pos"]
    source_root_quat = source_motion["root_quat"]
    source_joint_pos = source_motion["joint_pos"]
    source_fps = float(source_motion["fps"])

    source_key_body_pos, _ = _build_body_kinematics(
        root_link,
        joints,
        source_root_pos,
        source_root_quat,
        source_joint_pos,
        body_names=source_motion["key_body_names"],
    )
    key_body_errors = []
    for key_index, key_name in enumerate(source_motion["key_body_names"]):
        key_body_errors.append(
            np.linalg.norm(source_key_body_pos[:, key_index] - source_motion["key_body_pos"][:, key_index])
        )
    key_body_fk_max_error = float(np.max(key_body_errors)) if key_body_errors else 0.0
    if key_body_fk_max_error > 1.0e-4:
        raise ValueError(
            f"{source}: source key-body positions disagree with the T800 URDF; "
            f"max error={key_body_fk_max_error:.8g} m"
        )

    source_times = np.arange(len(source_joint_pos), dtype=np.float64) / source_fps
    target_times = _output_times(len(source_joint_pos), source_fps, output_fps)
    root_pos = _linear_interpolate(source_root_pos, source_times, target_times)
    root_quat = _slerp_interpolate(source_root_quat, source_times, target_times)
    joint_pos = _linear_interpolate(source_joint_pos, source_times, target_times)
    body_pos, body_quat = _build_body_kinematics(root_link, joints, root_pos, root_quat, joint_pos)

    dt = 1.0 / output_fps
    joint_vel = np.gradient(joint_pos, dt, axis=0)
    body_lin_vel = np.gradient(body_pos, dt, axis=0)
    body_ang_vel = _angular_velocity(body_quat, dt)

    arrays = {
        "fps": np.asarray([output_fps], dtype=np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin_vel.astype(np.float32),
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
        "body_names": np.asarray(CANONICAL_BODY_NAMES),
        "body_schema_version": np.asarray([1], dtype=np.int64),
        "source_file": np.asarray(source.name),
        "source_sha256": np.asarray(hashlib.sha256(source.read_bytes()).hexdigest()),
        "source_fps": np.asarray([source_fps], dtype=np.float32),
        "source_frame_count": np.asarray([len(source_joint_pos)], dtype=np.int64),
        "source_duration_s": np.asarray([(len(source_joint_pos) - 1) / source_fps], dtype=np.float32),
        "source_loop_mode": np.asarray([source_motion["loop_mode"]], dtype=np.int64),
        "source_key_body_fk_max_error_m": np.asarray([key_body_fk_max_error], dtype=np.float32),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    output_duration = (len(joint_pos) - 1) / output_fps
    print(
        f"[OK] {source.name} -> {output} | frames={len(joint_pos)} | "
        f"source={source_fps:g} Hz/{(len(source_joint_pos) - 1) / source_fps:.3f}s | "
        f"output={output_fps:g} Hz/{output_duration:.3f}s | "
        f"key_body_fk_max={key_body_fk_max_error:.3e} m"
    )
    return len(joint_pos), output_duration, key_body_fk_max_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "motions",
        nargs="*",
        type=Path,
        help="Input .pkl files; omit them to convert every .pkl in --source-dir.",
    )
    parser.add_argument("--source-dir", "--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--fps", type=float, default=DEFAULT_OUTPUT_FPS, help="Uniform output motion rate.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output NPZ files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0.0:
        raise ValueError(f"Output fps must be positive, got {args.fps}")
    if not args.urdf.is_file():
        raise FileNotFoundError(f"T800 URDF not found: {args.urdf}")

    motions = args.motions or sorted(args.source_dir.glob("*.pkl"))
    if not motions:
        raise FileNotFoundError(f"No .pkl motions found in {args.source_dir}")

    root_link, joints = load_urdf_tree(args.urdf)
    if root_link != "LINK_BASE":
        raise ValueError(f"Expected T800 root LINK_BASE, got {root_link}")
    for motion in motions:
        if not motion.is_file():
            raise FileNotFoundError(f"Input motion not found: {motion}")
        output = args.output_dir / f"{motion.stem}.npz"
        convert_motion(motion, output, root_link, joints, args.fps, args.overwrite)


if __name__ == "__main__":
    main()
