#!/usr/bin/env python3
"""Build non-destructive canonical motion variants for the four improved T800 tasks."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


POSITION_KEYS = ("joint_pos", "body_pos_w", "body_quat_w")
VELOCITY_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")
REQUIRED_KEYS = (*POSITION_KEYS, *VELOCITY_KEYS, "fps", "body_names", "body_schema_version")
DEFAULT_URDF = Path("source/whole_body_tracking/whole_body_tracking/assets/t800/urdf/serial_t800.urdf")
SPECS = {
    "straight_punch_L.npz": ("straight_punch_L_terminal_hold_0p5s.npz", 0.5, None),
    "straight_punch_R.npz": ("straight_punch_R_terminal_hold_0p5s.npz", 0.5, None),
    "left_hook_001.npz": ("left_hook_001_terminal_hold_0p5s.npz", 0.5, None),
    # Slow the physically infeasible hip-speed spike around source frames 52-61.
    "roundhouse_kick_001.npz": (
        "roundhouse_kick_001_retimed_terminal_hold_1s.npz",
        1.0,
        (52.0, 61.0, 0.5),
    ),
}


def _load_canonical_motion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as motion:
        missing = [key for key in REQUIRED_KEYS if key not in motion]
        if missing:
            raise ValueError(f"{path}: missing canonical arrays {missing}")
        arrays = {key: np.asarray(motion[key]) for key in motion.files}

    schema_version = int(np.asarray(arrays["body_schema_version"]).reshape(-1)[0])
    body_names = tuple(str(name) for name in arrays["body_names"].tolist())
    if schema_version != 1 or len(body_names) != 34:
        raise ValueError(f"{path}: expected canonical T800 body schema v1 with 34 bodies")
    return arrays


def append_terminal_hold(arrays: dict[str, np.ndarray], hold_seconds: float) -> dict[str, np.ndarray]:
    fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
    hold_frames = int(round(fps * hold_seconds))
    if fps <= 0.0 or hold_frames <= 0:
        raise ValueError(f"Invalid fps/hold: fps={fps}, hold_seconds={hold_seconds}")

    result: dict[str, np.ndarray] = {}
    for key, values in arrays.items():
        values = np.asarray(values)
        if key in POSITION_KEYS:
            tail = np.repeat(values[-1:], hold_frames, axis=0)
            result[key] = np.concatenate((values, tail), axis=0)
        elif key in VELOCITY_KEYS:
            tail = np.zeros((hold_frames, *values.shape[1:]), dtype=values.dtype)
            result[key] = np.concatenate((values, tail), axis=0)
        else:
            result[key] = values.copy()

    result["derived_terminal_hold_seconds"] = np.asarray([hold_seconds], dtype=np.float32)
    result["derived_from_canonical"] = np.asarray([True])
    return result


def _linear_interpolate(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    source_frames = np.arange(values.shape[0], dtype=np.float64)
    output = np.empty((len(coordinates), flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(coordinates, source_frames, flat[:, column])
    return output.reshape(len(coordinates), *values.shape[1:]).astype(values.dtype)


def _quaternion_nlerp(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, values.shape[0] - 1)
    alpha = (coordinates - lower).reshape(-1, 1, 1)
    q0 = values[lower].astype(np.float64)
    q1 = values[upper].astype(np.float64)
    q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0.0, -q1, q1)
    output = (1.0 - alpha) * q0 + alpha * q1
    output /= np.linalg.norm(output, axis=-1, keepdims=True).clip(min=1.0e-12)
    return output.astype(values.dtype)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.asarray(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.asarray([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale])
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.asarray([(matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale])
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.asarray([(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])
    return quaternion / np.linalg.norm(quaternion)


def _load_urdf_joints(path: Path) -> tuple[str, list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray | None, int | None]]]:
    robot = ET.parse(path).getroot()
    links = {element.attrib["name"] for element in robot.findall("link")}
    children: set[str] = set()
    joints = []
    motion_index = 0
    for element in robot.findall("joint"):
        parent = element.find("parent").attrib["link"]
        child = element.find("child").attrib["link"]
        children.add(child)
        origin = element.find("origin")
        xyz = np.fromstring("0 0 0" if origin is None else origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring("0 0 0" if origin is None else origin.attrib.get("rpy", "0 0 0"), sep=" ")
        if element.attrib["type"] == "fixed":
            axis, index = None, None
        else:
            axis_element = element.find("axis")
            axis = np.fromstring("1 0 0" if axis_element is None else axis_element.attrib.get("xyz", "1 0 0"), sep=" ")
            axis /= np.linalg.norm(axis)
            index = motion_index
            motion_index += 1
        joints.append((parent, child, xyz, _rpy_matrix(rpy), axis, index))
    roots = links - children
    if len(roots) != 1 or motion_index != 25:
        raise ValueError(f"Unexpected T800 URDF structure: roots={sorted(roots)}, joints={motion_index}")
    return roots.pop(), joints


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _angular_velocity(quaternions: np.ndarray, dt: float) -> np.ndarray:
    interval = np.empty((len(quaternions) - 1, quaternions.shape[1], 3), dtype=np.float64)
    for frame in range(len(quaternions) - 1):
        for body in range(quaternions.shape[1]):
            q0 = quaternions[frame, body].astype(np.float64)
            q1 = quaternions[frame + 1, body].astype(np.float64)
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
    output[0], output[-1] = interval[0], interval[-1]
    output[1:-1] = 0.5 * (interval[:-1] + interval[1:])
    return output.astype(quaternions.dtype)


def rebuild_body_kinematics(arrays: dict[str, np.ndarray], urdf_path: Path) -> dict[str, np.ndarray]:
    root_link, joints = _load_urdf_joints(urdf_path)
    body_names = tuple(str(name) for name in arrays["body_names"].tolist())
    aliases = {
        "LINK_ANKLE_ROLL_L_TOE": "LINK_ANKLE_ROLL_L",
        "LINK_ANKLE_ROLL_L_HEEL": "LINK_ANKLE_ROLL_L",
        "LINK_ANKLE_ROLL_R_TOE": "LINK_ANKLE_ROLL_R",
        "LINK_ANKLE_ROLL_R_HEEL": "LINK_ANKLE_ROLL_R",
        "LINK_WRIST_PITCH_L": "LINK_ELBOW_YAW_L",
        "LINK_WRIST_ROLL_L": "LINK_ELBOW_YAW_L",
        "LINK_WRIST_PITCH_R": "LINK_ELBOW_YAW_R",
        "LINK_WRIST_ROLL_R": "LINK_ELBOW_YAW_R",
    }
    root_index = body_names.index(root_link)
    positions = np.empty_like(arrays["body_pos_w"])
    quaternions = np.empty_like(arrays["body_quat_w"])

    for frame in range(len(arrays["joint_pos"])):
        transforms = {
            root_link: (
                arrays["body_pos_w"][frame, root_index].astype(np.float64),
                _quaternion_wxyz_to_matrix(arrays["body_quat_w"][frame, root_index]),
            )
        }
        unresolved = list(joints)
        while unresolved:
            remaining = []
            for parent, child, origin_position, origin_rotation, axis, joint_index in unresolved:
                if parent not in transforms:
                    remaining.append((parent, child, origin_position, origin_rotation, axis, joint_index))
                    continue
                parent_position, parent_rotation = transforms[parent]
                child_position = parent_position + parent_rotation @ origin_position
                child_rotation = parent_rotation @ origin_rotation
                if joint_index is not None:
                    child_rotation = child_rotation @ _axis_angle_matrix(axis, float(arrays["joint_pos"][frame, joint_index]))
                transforms[child] = (child_position, child_rotation)
            if len(remaining) == len(unresolved):
                raise ValueError("T800 URDF joint graph is disconnected")
            unresolved = remaining

        for body_index, body_name in enumerate(body_names):
            position, rotation = transforms[aliases.get(body_name, body_name)]
            positions[frame, body_index] = position
            quaternion = _matrix_to_quaternion_wxyz(rotation)
            if frame > 0 and np.dot(quaternion, quaternions[frame - 1, body_index]) < 0.0:
                quaternion = -quaternion
            quaternions[frame, body_index] = quaternion

    dt = 1.0 / float(np.asarray(arrays["fps"]).reshape(-1)[0])
    result = {key: np.array(value, copy=True) for key, value in arrays.items()}
    result["body_pos_w"] = positions
    result["body_quat_w"] = quaternions
    result["joint_vel"] = np.gradient(result["joint_pos"], dt, axis=0).astype(result["joint_pos"].dtype)
    result["body_lin_vel_w"] = np.gradient(positions, dt, axis=0).astype(positions.dtype)
    result["body_ang_vel_w"] = _angular_velocity(quaternions, dt)
    return result


def retime_local_segment(
    arrays: dict[str, np.ndarray],
    source_start: float,
    source_end: float,
    speed_scale: float,
) -> dict[str, np.ndarray]:
    if not (0.0 < speed_scale < 1.0 and 0.0 <= source_start < source_end):
        raise ValueError(
            f"Invalid local retime: start={source_start}, end={source_end}, speed_scale={speed_scale}"
        )

    frame_count = int(arrays["joint_pos"].shape[0])
    coordinates = [0.0]
    while coordinates[-1] < frame_count - 1:
        current = coordinates[-1]
        step = speed_scale if source_start <= current < source_end else 1.0
        coordinates.append(min(current + step, float(frame_count - 1)))
    coordinates_array = np.asarray(coordinates, dtype=np.float64)
    source_rate = np.gradient(coordinates_array)

    result = {key: np.array(value, copy=True) for key, value in arrays.items()}
    result["joint_pos"] = _linear_interpolate(arrays["joint_pos"], coordinates_array)
    result["body_pos_w"] = _linear_interpolate(arrays["body_pos_w"], coordinates_array)
    result["body_quat_w"] = _quaternion_nlerp(arrays["body_quat_w"], coordinates_array)
    for key in VELOCITY_KEYS:
        interpolated = _linear_interpolate(arrays[key], coordinates_array)
        scale_shape = (len(source_rate),) + (1,) * (interpolated.ndim - 1)
        result[key] = (interpolated * source_rate.reshape(scale_shape)).astype(interpolated.dtype)

    result["derived_source_frame_coordinates"] = coordinates_array.astype(np.float32)
    result["derived_local_slowdown"] = np.asarray([source_start, source_end, speed_scale], dtype=np.float32)
    return result


def remove_planar_foot_midpoint_drift(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove only global XY drift while preserving pose and foot separation."""

    body_names = tuple(str(name) for name in arrays["body_names"].tolist())
    left_foot_index = body_names.index("LINK_ANKLE_ROLL_L")
    right_foot_index = body_names.index("LINK_ANKLE_ROLL_R")
    foot_midpoint_xy = arrays["body_pos_w"][:, [left_foot_index, right_foot_index], :2].mean(axis=1)
    drift_xy = foot_midpoint_xy - foot_midpoint_xy[0]
    drift_velocity_xy = np.gradient(drift_xy, 1.0 / float(np.asarray(arrays["fps"]).reshape(-1)[0]), axis=0)

    result = {key: np.array(value, copy=True) for key, value in arrays.items()}
    result["body_pos_w"][:, :, :2] -= drift_xy[:, None, :]
    result["body_lin_vel_w"][:, :, :2] -= drift_velocity_xy[:, None, :]
    result["derived_removed_foot_midpoint_drift_xy"] = drift_xy.astype(np.float32)
    return result


def prepare_motion(
    source: Path,
    output: Path,
    hold_seconds: float,
    local_slowdown: tuple[float, float, float] | None = None,
    urdf_path: Path = DEFAULT_URDF,
    overwrite: bool = False,
) -> tuple[int, int, float]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {output}")

    arrays = _load_canonical_motion(source)
    if source.name == "left_hook_001.npz":
        arrays = remove_planar_foot_midpoint_drift(arrays)
    if local_slowdown is not None:
        arrays = retime_local_segment(arrays, *local_slowdown)
        arrays = rebuild_body_kinematics(arrays, urdf_path)
    result = append_terminal_hold(arrays, hold_seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, **result)

    source_frames = int(arrays["joint_pos"].shape[0])
    output_frames = int(result["joint_pos"].shape[0])
    fps = float(np.asarray(result["fps"]).reshape(-1)[0])
    return source_frames, output_frames, fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, default=Path("data/npz/traj_eng_50hz_canonical"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/npz/traj_eng_50hz_improved"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.canonical_dir.resolve() == args.output_dir.resolve():
        raise ValueError("--output-dir must differ from --canonical-dir")

    for source_name, (output_name, hold_seconds, local_slowdown) in SPECS.items():
        source = args.canonical_dir / source_name
        output = args.output_dir / output_name
        old_frames, new_frames, fps = prepare_motion(
            source,
            output,
            hold_seconds,
            local_slowdown=local_slowdown,
            overwrite=args.overwrite,
        )
        print(
            f"[OK] {output}: {old_frames} + {new_frames - old_frames} = {new_frames} frames, "
            f"episode={new_frames / fps:.2f}s at {fps:g} Hz"
        )


if __name__ == "__main__":
    main()
