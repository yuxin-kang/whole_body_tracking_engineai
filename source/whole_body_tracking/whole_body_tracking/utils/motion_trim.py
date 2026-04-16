from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


TIME_SERIES_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def load_motion_npz(path: str | Path) -> dict[str, np.ndarray]:
    motion_path = Path(path)
    if motion_path.suffix.lower() != ".npz":
        raise ValueError(f"Expected a .npz motion file, got: {motion_path}")
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    with np.load(motion_path) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


def infer_total_frames(data: Mapping[str, np.ndarray]) -> int:
    frame_count = _infer_frame_count(data)

    for key in TIME_SERIES_KEYS:
        if key not in data:
            continue
        array = data[key]
        if _time_axis_length(array, key=key) != frame_count:
            raise ValueError(
                f"Inconsistent time-series leading dimensions: '{key}' has {array.shape[0]} frames, "
                f"expected {frame_count}."
            )

    return frame_count


def trim_motion_dict(data: Mapping[str, np.ndarray], start_frame: int, end_frame: int) -> dict[str, np.ndarray]:
    total_frames = infer_total_frames(data)
    _validate_trim_range(start_frame, end_frame, total_frames)

    trimmed: dict[str, np.ndarray] = {}
    for key, value in data.items():
        if not isinstance(value, np.ndarray):
            trimmed[key] = np.asarray(value)
            continue

        if value.ndim > 0 and value.shape[0] == total_frames:
            trimmed[key] = np.array(value[start_frame : end_frame + 1], copy=True)
        else:
            trimmed[key] = np.array(value, copy=True)

    return trimmed


def default_trim_output_path(input_path: str | Path, start_frame: int, end_frame: int) -> Path:
    path = Path(input_path)
    return path.with_name(f"{path.stem}_trim_{start_frame:04d}_{end_frame:04d}{path.suffix}")


def save_trimmed_motion_npz(data: Mapping[str, np.ndarray], output_path: str | Path, force: bool = False) -> Path:
    path = Path(output_path)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing output file: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return path


def _time_axis_length(array: np.ndarray, key: str) -> int:
    if array.ndim == 0:
        raise ValueError(f"Motion key '{key}' must have a time dimension.")
    if array.shape[0] <= 0:
        raise ValueError(f"Motion key '{key}' must contain at least one frame.")
    return int(array.shape[0])


def _infer_frame_count(data: Mapping[str, np.ndarray]) -> int:
    for key in TIME_SERIES_KEYS:
        if key in data:
            return _time_axis_length(data[key], key=key)

    frame_candidates = [int(value.shape[0]) for value in data.values() if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] > 1]
    if not frame_candidates:
        raise ValueError("Unable to infer motion frame count from the provided arrays.")
    return frame_candidates[0]


def _validate_trim_range(start_frame: int, end_frame: int, total_frames: int) -> None:
    if start_frame < 0 or end_frame < 0:
        raise ValueError("Trim markers must be non-negative.")
    if start_frame > end_frame:
        raise ValueError("Trim start_frame must be less than or equal to end_frame.")
    if end_frame >= total_frames:
        raise ValueError(f"Trim markers must stay within 0..{total_frames - 1}.")
