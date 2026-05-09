"""Plot the T800 540 right-ankle relative height curve with frame annotations.

Example:
    python scripts/plot_540_right_ankle_curve.py \
        --input_file data/npz/540/cut/540huixuantitui1.npz
"""

from __future__ import annotations

import argparse
import sys
from importlib import util
from pathlib import Path


def _load_motion_landmarks_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "utils"
        / "motion_landmarks.py"
    )
    spec = util.spec_from_file_location("motion_landmarks", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Plot the T800 540 right-ankle height curve.")
    parser.add_argument("--input_file", required=True, help="Path to a local .npz motion file.")
    parser.add_argument(
        "--output_file",
        default="outputs/2026-05-01/540_right_ankle_height_curve.png",
        help="Path to save the generated PNG plot.",
    )
    args = parser.parse_args()

    module = _load_motion_landmarks_module()
    output_file = module.save_t800_right_ankle_height_plot(args.input_file, args.output_file)
    report = module.summarize_t800_right_ankle_540_landmarks(args.input_file)

    print(report.format())
    print(f"[INFO] Saved right-ankle height plot: {output_file}")


if __name__ == "__main__":
    main()
