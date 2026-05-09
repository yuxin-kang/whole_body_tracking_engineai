from pathlib import Path

import numpy as np


ZERO_Z_THRESHOLD = 0.015
MIN_QISHI_ZERO_RUN = 20
MIN_RETURN_ZERO_RUN = 3


def _t800_motion_body_names() -> list[str]:
    # Mirrored from t800_mdp.py so the test can stay import-light.
    return [
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


def _right_ankle_relative_z() -> np.ndarray:
    motion_path = Path("data/npz/540/cut/540huixuantitui1.npz")
    motion = np.load(motion_path)
    body_names = _t800_motion_body_names()
    ankle_index = body_names.index("LINK_ANKLE_ROLL_R")
    z_values = motion["body_pos_w"][:, ankle_index, 2]
    return z_values - z_values[0]


def _extract_right_ankle_qishi_peak_return_landmarks() -> tuple[int, int, int, int]:
    relative_z = _right_ankle_relative_z()
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
    return qishi_start, qishi_end, peak_frame, return_zero_end


def _extract_right_ankle_local_min_peak_landmarks() -> tuple[int, int, int]:
    relative_z = _right_ankle_relative_z()
    peak_frame = int(np.argmax(relative_z))
    local_minima = _find_local_minima(relative_z)
    pre_peak_candidates = local_minima[local_minima < peak_frame]
    post_peak_candidates = local_minima[local_minima > peak_frame]

    pre_peak_frame = int(pre_peak_candidates[-1])
    post_peak_frame = int(post_peak_candidates[0])
    return pre_peak_frame, peak_frame, post_peak_frame


def test_540_tui_motion_right_ankle_z_landmarks():
    qishi_start, qishi_end, peak_frame, return_zero_end = _extract_right_ankle_qishi_peak_return_landmarks()

    print(
        "540 right ankle relative-z landmarks:"
        f" qishi_start={qishi_start},"
        f" qishi_end={qishi_end},"
        f" peak_frame={peak_frame},"
        f" return_zero_end={return_zero_end}"
    )

    assert (qishi_start, qishi_end, peak_frame, return_zero_end) == (0, 111, 155, 202)


def test_540_tui_motion_right_ankle_peak_local_min_landmarks():
    pre_peak_frame, peak_frame, post_peak_frame = _extract_right_ankle_local_min_peak_landmarks()

    print(
        "540 right ankle local-min landmarks:"
        f" pre_peak_frame={pre_peak_frame},"
        f" peak_frame={peak_frame},"
        f" post_peak_frame={post_peak_frame}"
    )

    assert (pre_peak_frame, peak_frame, post_peak_frame) == (134, 155, 170)


def test_540_config_phase_windows_follow_right_ankle_relative_z_landmarks():
    config = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py").read_text()
    frame_count = np.load("data/npz/540/cut/540huixuantitui1.npz")["body_pos_w"].shape[0]
    denom = frame_count - 1
    _, qishi_end, peak_frame, return_zero_end = _extract_right_ankle_qishi_peak_return_landmarks()

    kick_phase = (qishi_end / denom, peak_frame / denom)
    retract_phase = (peak_frame / denom, return_zero_end / denom)

    assert f"T800_540_KICK_PHASE = ({kick_phase[0]:.4f}, {kick_phase[1]:.4f})" in config
    assert f"T800_540_RETRACT_PHASE = ({retract_phase[0]:.4f}, {retract_phase[1]:.4f})" in config


def test_540_config_late_phase_windows_follow_right_ankle_peak_local_min_landmarks():
    config = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py").read_text()
    frame_count = np.load("data/npz/540/cut/540huixuantitui1.npz")["body_pos_w"].shape[0]
    denom = frame_count - 1
    pre_peak_frame, peak_frame, post_peak_frame = _extract_right_ankle_local_min_peak_landmarks()

    kick_phase_late = (pre_peak_frame / denom, peak_frame / denom)
    retract_phase_late = (peak_frame / denom, post_peak_frame / denom)

    assert f"T800_540_KICK_PHASE_LATE = ({kick_phase_late[0]:.4f}, {kick_phase_late[1]:.4f})" in config
    assert f"T800_540_RETRACT_PHASE_LATE = ({retract_phase_late[0]:.4f}, {retract_phase_late[1]:.4f})" in config


def test_540_config_pre_kick_phase_starts_from_first_grounded_frame_before_peak():
    config = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py").read_text()
    frame_count = np.load("data/npz/540/cut/540huixuantitui1.npz")["body_pos_w"].shape[0]
    denom = frame_count - 1
    qishi_start, _, peak_frame, _ = _extract_right_ankle_qishi_peak_return_landmarks()

    pre_kick_phase = (qishi_start / denom, peak_frame / denom)

    assert f"T800_540_PRE_KICK_PHASE = ({pre_kick_phase[0]:.4f}, {pre_kick_phase[1]:.4f})" in config
