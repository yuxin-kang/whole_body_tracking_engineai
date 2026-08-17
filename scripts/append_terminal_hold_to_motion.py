#!/usr/bin/env python3
"""Append a stationary terminal hold to a T800 tracking motion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


POSITION_KEYS = ("joint_pos", "body_pos_w", "body_quat_w")
VELOCITY_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")
REQUIRED_KEYS = POSITION_KEYS + VELOCITY_KEYS + ("fps",)


def append_terminal_hold(source: Path, output: Path, hold_seconds: float) -> tuple[int, int, float]:
    with np.load(source, allow_pickle=False) as motion:
        missing = [key for key in REQUIRED_KEYS if key not in motion]
        if missing:
            raise ValueError(f"Motion is missing required arrays {missing}: {source}")

        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        if fps <= 0.0 or hold_seconds <= 0.0:
            raise ValueError(f"fps and hold_seconds must be positive, got fps={fps}, hold={hold_seconds}")

        frame_count = int(motion["joint_pos"].shape[0])
        hold_frames = int(round(hold_seconds * fps))
        if hold_frames <= 0:
            raise ValueError(f"hold_seconds produces no frames at {fps:g} Hz: {hold_seconds}")

        result: dict[str, np.ndarray] = {}
        for key in motion.files:
            values = np.asarray(motion[key])
            if key in POSITION_KEYS:
                tail = np.repeat(values[-1:], hold_frames, axis=0)
                result[key] = np.concatenate((values, tail), axis=0)
            elif key in VELOCITY_KEYS:
                tail = np.zeros((hold_frames, *values.shape[1:]), dtype=values.dtype)
                result[key] = np.concatenate((values, tail), axis=0)
            else:
                result[key] = values.copy()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    return frame_count, frame_count + hold_frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    args = parser.parse_args()

    old_frames, new_frames, fps = append_terminal_hold(args.source, args.output, args.hold_seconds)
    print(
        f"Wrote {args.output}: {old_frames} + {new_frames - old_frames} = {new_frames} frames, "
        f"episode={new_frames / fps:.2f}s at {fps:g} Hz"
    )


if __name__ == "__main__":
    main()
