from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path

import numpy as np

T800_MOTION_BODY_NAMES = [
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
]

ZERO_Z_THRESHOLD = 0.015
MIN_QISHI_ZERO_RUN = 20
MIN_RETURN_ZERO_RUN = 3
RIGHT_ANKLE_BODY_NAME = "LINK_ANKLE_ROLL_R"


@dataclass(frozen=True)
class T800RightAnkleLandmarkReport:
    motion_file: str
    fps: float
    frame_count: int
    qishi_start: int
    qishi_end: int
    peak_frame: int
    return_zero_end: int
    pre_peak_local_min: int
    post_peak_local_min: int
    kick_phase: tuple[float, float]
    retract_phase: tuple[float, float]
    kick_phase_late: tuple[float, float]
    retract_phase_late: tuple[float, float]

    def format(self) -> str:
        return "\n".join(
            [
                "[INFO] 540 right ankle landmark report",
                f"  motion_file: {self.motion_file}",
                f"  fps: {self.fps:.4f}",
                f"  frame_count: {self.frame_count}",
                f"  qishi_start: {self.qishi_start}",
                f"  qishi_end: {self.qishi_end}",
                f"  peak_frame: {self.peak_frame}",
                f"  return_zero_end: {self.return_zero_end}",
                f"  pre_peak_local_min: {self.pre_peak_local_min}",
                f"  post_peak_local_min: {self.post_peak_local_min}",
                f"  kick_phase: ({self.kick_phase[0]:.4f}, {self.kick_phase[1]:.4f})",
                f"  retract_phase: ({self.retract_phase[0]:.4f}, {self.retract_phase[1]:.4f})",
                f"  kick_phase_late: ({self.kick_phase_late[0]:.4f}, {self.kick_phase_late[1]:.4f})",
                f"  retract_phase_late: ({self.retract_phase_late[0]:.4f}, {self.retract_phase_late[1]:.4f})",
            ]
        )


def load_t800_right_ankle_relative_height(motion_file: str) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(motion_file) as motion:
        body_names: Sequence[str] = T800_MOTION_BODY_NAMES
        ankle_index = body_names.index(RIGHT_ANKLE_BODY_NAME)
        frame_ids = np.arange(motion["body_pos_w"].shape[0], dtype=np.int32)
        relative_z = motion["body_pos_w"][:, ankle_index, 2] - motion["body_pos_w"][0, ankle_index, 2]
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    return frame_ids, relative_z, fps


def _find_local_minima(signal: np.ndarray) -> np.ndarray:
    left = signal[1:-1] <= signal[:-2]
    right = signal[1:-1] < signal[2:]
    return np.flatnonzero(left & right) + 1


def _find_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    run_start: int | None = None

    for index, is_true in enumerate(mask):
        if is_true and run_start is None:
            run_start = index
        elif not is_true and run_start is not None:
            runs.append((run_start, index - 1))
            run_start = None

    if run_start is not None:
        runs.append((run_start, len(mask) - 1))

    return runs


def summarize_t800_right_ankle_540_landmarks(motion_file: str) -> T800RightAnkleLandmarkReport:
    frame_ids, relative_z, fps = load_t800_right_ankle_relative_height(motion_file)
    with np.load(motion_file) as motion:
        peak_frame = int(np.argmax(relative_z))
        zero_runs = _find_true_runs(np.abs(relative_z) <= ZERO_Z_THRESHOLD)
        pre_peak_runs = [
            run
            for run in zero_runs
            if run[1] < peak_frame and (run[1] - run[0] + 1) >= MIN_QISHI_ZERO_RUN
        ]
        post_peak_runs = [
            run
            for run in zero_runs
            if run[0] > peak_frame and (run[1] - run[0] + 1) >= MIN_RETURN_ZERO_RUN
        ]

        qishi_start, qishi_end = max(pre_peak_runs, key=lambda run: (run[1] - run[0], run[1]))
        _, return_zero_end = post_peak_runs[0]

        local_minima = _find_local_minima(relative_z)
        pre_peak_candidates = local_minima[local_minima < peak_frame]
        post_peak_candidates = local_minima[local_minima > peak_frame]
        pre_peak_local_min = int(pre_peak_candidates[-1])
        post_peak_local_min = int(post_peak_candidates[0])

        frame_count = int(frame_ids.shape[0])

    denom = max(frame_count - 1, 1)
    kick_phase = (qishi_end / denom, peak_frame / denom)
    retract_phase = (peak_frame / denom, return_zero_end / denom)
    kick_phase_late = (pre_peak_local_min / denom, peak_frame / denom)
    retract_phase_late = (peak_frame / denom, post_peak_local_min / denom)

    return T800RightAnkleLandmarkReport(
        motion_file=motion_file,
        fps=fps,
        frame_count=frame_count,
        qishi_start=int(qishi_start),
        qishi_end=int(qishi_end),
        peak_frame=peak_frame,
        return_zero_end=int(return_zero_end),
        pre_peak_local_min=pre_peak_local_min,
        post_peak_local_min=post_peak_local_min,
        kick_phase=kick_phase,
        retract_phase=retract_phase,
        kick_phase_late=kick_phase_late,
        retract_phase_late=retract_phase_late,
    )


def save_t800_right_ankle_height_plot(motion_file: str, output_file: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_ids, relative_z, fps = load_t800_right_ankle_relative_height(motion_file)
    report = summarize_t800_right_ankle_540_landmarks(motion_file)
    keyframes = {
        report.qishi_end: ("qishi_end", "#1f77b4"),
        report.pre_peak_local_min: ("pre_peak_local_min", "#ff7f0e"),
        report.peak_frame: ("peak_frame", "#d62728"),
        report.post_peak_local_min: ("post_peak_local_min", "#2ca02c"),
        report.return_zero_end: ("return_zero_end", "#9467bd"),
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(36, 12), dpi=200)
    ax.plot(frame_ids, relative_z, color="#264653", linewidth=1.4, marker="o", markersize=2.8)

    for frame_id, height in zip(frame_ids, relative_z):
        ax.annotate(
            str(int(frame_id)),
            (int(frame_id), float(height)),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=4,
            color="#4a4a4a",
        )

    for frame_id, (label, color) in keyframes.items():
        height = float(relative_z[frame_id])
        ax.scatter([frame_id], [height], color=color, s=80, zorder=3)
        ax.axvline(frame_id, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.annotate(
            f"{label}={frame_id}",
            (frame_id, height),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.9},
        )

    ax.set_title(
        f"T800 Right Ankle Relative Height\n{Path(motion_file).name} | fps={fps:.4f} | frames={frame_ids.shape[0]}",
        fontsize=16,
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Relative Height (m)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
