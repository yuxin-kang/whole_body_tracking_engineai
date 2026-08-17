#!/usr/bin/env python3
"""Canonicalize legacy traj_eng T800 motion body tensors without touching source files.

The six official ``traj_eng_50hz`` motions use a different 34-body axis order
than the repository's current ``T800_MOTION_BODY_NAMES`` contract.  Joint
tensors are already in the expected DFS order and must not be changed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np


BODY_TENSOR_KEYS = (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
REQUIRED_KEYS = ("joint_pos", "joint_vel", *BODY_TENSOR_KEYS, "fps")

# Canonical body names used by t800_mdp.T800_MOTION_BODY_NAMES.  This script is
# deliberately IsaacLab-free so data repair and validation can run on a CPU
# login node.  A unit test locks this list to the runtime constant.
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

# canonical body index -> legacy traj_eng body index
#
# The current T800 converter represents unavailable toe/heel bodies with the
# ankle-roll body and unavailable wrist bodies with the elbow-yaw body.  The
# repeated indexes below reproduce that same canonical contract exactly.
CANONICAL_FROM_TRAJ_ENG = np.asarray(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        6,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        13,
        13,
        15,
        16,
        17,
        18,
        19,
        20,
        20,
        20,
        24,
        25,
        26,
        27,
        28,
        28,
        28,
        32,
        33,
    ],
    dtype=np.int64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _max_body_delta(body_tensor: np.ndarray, first: int, second: int) -> float:
    return float(np.max(np.abs(body_tensor[:, first] - body_tensor[:, second])))


def validate_traj_eng_source(arrays: dict[str, np.ndarray], source: Path | str = "<memory>") -> None:
    """Reject non-traj_eng or already-canonical tensors before remapping."""

    missing = [key for key in REQUIRED_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"{source}: missing required arrays: {missing}")

    frame_count = arrays["joint_pos"].shape[0]
    if arrays["joint_pos"].ndim != 2 or arrays["joint_pos"].shape[1] != 25:
        raise ValueError(f"{source}: expected joint_pos shape (frames, 25), got {arrays['joint_pos'].shape}")
    if arrays["joint_vel"].shape != arrays["joint_pos"].shape:
        raise ValueError(
            f"{source}: joint_vel shape {arrays['joint_vel'].shape} does not match "
            f"joint_pos {arrays['joint_pos'].shape}"
        )

    expected_tail_shapes = {
        "body_pos_w": (34, 3),
        "body_quat_w": (34, 4),
        "body_lin_vel_w": (34, 3),
        "body_ang_vel_w": (34, 3),
    }
    for key, tail_shape in expected_tail_shapes.items():
        expected_shape = (frame_count, *tail_shape)
        if arrays[key].shape != expected_shape:
            raise ValueError(f"{source}: expected {key} shape {expected_shape}, got {arrays[key].shape}")

    if "body_names" in arrays:
        body_names = tuple(str(name) for name in arrays["body_names"].tolist())
        if body_names == CANONICAL_BODY_NAMES:
            raise ValueError(f"{source}: input already appears to use the canonical T800 body order")

    # Legacy traj_eng has one repeated ankle body per leg.  Canonical data has
    # three repeated entries (ankle + toe + heel).  These checks distinguish
    # the two schemas and prevent applying the remap twice.
    body_pos = arrays["body_pos_w"]
    tolerance = 1.0e-6
    legacy_pairs = ((6, 7), (13, 14))
    canonical_extra_pairs = ((6, 8), (14, 15), (14, 16))
    if any(_max_body_delta(body_pos, a, b) > tolerance for a, b in legacy_pairs):
        raise ValueError(f"{source}: body axis does not match the legacy traj_eng ankle signature")
    if all(_max_body_delta(body_pos, a, b) <= tolerance for a, b in canonical_extra_pairs):
        raise ValueError(f"{source}: input already appears to use the canonical T800 body order")


def remap_motion_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a canonical copy while preserving every non-body tensor."""

    result = {key: np.array(value, copy=True) for key, value in arrays.items()}
    for key in BODY_TENSOR_KEYS:
        result[key] = np.asarray(arrays[key])[:, CANONICAL_FROM_TRAJ_ENG].copy()

    result["body_names"] = np.asarray(CANONICAL_BODY_NAMES)
    result["body_schema_version"] = np.asarray([1], dtype=np.int32)
    result["source_body_schema"] = np.asarray(["t800_traj_eng_legacy_v0"])
    return result


def validate_canonical_result(arrays: dict[str, np.ndarray], source: Path | str = "<memory>") -> None:
    if tuple(str(name) for name in arrays["body_names"].tolist()) != CANONICAL_BODY_NAMES:
        raise ValueError(f"{source}: body_names metadata does not match the canonical T800 contract")

    tolerance = 1.0e-6
    alias_groups = ((6, 7, 8), (14, 15, 16), (22, 23, 24), (29, 30, 31))
    for key in BODY_TENSOR_KEYS:
        tensor = arrays[key]
        for group in alias_groups:
            if any(_max_body_delta(tensor, group[0], index) > tolerance for index in group[1:]):
                raise ValueError(f"{source}: {key} alias group {group} is not canonical")


def convert_file(source: Path, destination: Path, overwrite: bool = False) -> str:
    if source.resolve() == destination.resolve():
        raise ValueError(f"Refusing to overwrite source motion: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")

    with np.load(source, allow_pickle=False) as motion:
        arrays = {key: np.asarray(motion[key]) for key in motion.files}
    validate_traj_eng_source(arrays, source)

    result = remap_motion_arrays(arrays)
    result["source_sha256"] = np.asarray([_sha256(source)])
    validate_canonical_result(result, destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as stream:
            np.savez_compressed(stream, **result)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    return str(result["source_sha256"][0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/npz/traj_eng_50hz"),
        help="Directory containing legacy traj_eng npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/npz/traj_eng_50hz_canonical"),
        help="New directory for canonical npz files. It must differ from --input-dir.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace files in the output directory only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("--output-dir must differ from --input-dir")

    source_files = sorted(input_dir.glob("*.npz"))
    if not source_files:
        raise FileNotFoundError(f"No npz files found in {input_dir}")

    for source in source_files:
        destination = output_dir / source.name
        source_hash = convert_file(source, destination, overwrite=args.overwrite)
        print(f"[OK] {source.name}: {source_hash} -> {destination}")


if __name__ == "__main__":
    main()
