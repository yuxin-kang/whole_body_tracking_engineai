"""Turn time-aligned get-up Play samples into CSV, plots and descriptive metrics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def normalized_quaternions(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Zero quaternion is not an orientation")
    return q / norm


def tilt_degrees(q):
    """Inclination of the base z axis from world vertical; quaternion order wxyz."""
    q = normalized_quaternions(q)
    return np.rad2deg(np.arccos(np.clip(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1, 1)))


def orientation_error_degrees(q, reference):
    dot = np.sum(normalized_quaternions(q) * normalized_quaternions(reference), axis=-1)
    return np.rad2deg(2 * np.arccos(np.clip(np.abs(dot), 0, 1)))


def first_sustained_index(mask, count):
    if count < 1:
        raise ValueError("count must be positive")
    length = 0
    for index, active in enumerate(mask):
        length = length + 1 if active else 0
        if length >= count:
            return index - count + 1
    return None


def analyze(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = json.loads((directory / "metadata.json").read_text())
    with np.load(directory / "rollout.npz", allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    dt = metadata["step_dt"]
    rewards = metadata["reward_names"]
    joint_names = metadata["joint_names"]
    knees = [joint_names.index("J03_KNEE_PITCH_L"), joint_names.index("J09_KNEE_PITCH_R")]
    foot_names = ["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]
    feet = [metadata["contact_body_names"].index(name) for name in foot_names]
    body_feet = [metadata["body_names"].index(name) for name in foot_names]
    report = {"run": metadata["run"], "checkpoint": metadata["checkpoint"], "episodes": []}
    csv_rows = []
    for episode in np.unique(data["episode"]):
        select = data["episode"] == episode
        d = {key: value[select] for key, value in data.items()}
        t = d["episode_step"] * dt
        ref_t = d["frame"] * dt
        height = d["anchor_pos"][:, 2]
        tilt = tilt_degrees(d["anchor_quat"])
        ref_tilt = tilt_degrees(d["ref_anchor_quat"])
        error = orientation_error_degrees(d["anchor_quat"], d["ref_anchor_quat"])
        xy_error = np.linalg.norm(d["anchor_pos"][:, :2] - d["ref_anchor_pos"][:, :2], axis=-1)
        angular_speed = np.linalg.norm(d["anchor_ang_vel"], axis=-1)
        ref_angular_speed = np.linalg.norm(d["ref_anchor_ang_vel"], axis=-1)
        linear_speed = np.linalg.norm(d["anchor_lin_vel"], axis=-1)
        ref_linear_speed = np.linalg.norm(d["ref_anchor_lin_vel"], axis=-1)
        feet_fz = d["contact_force"][:, feet, 2]
        foot_speed = np.linalg.norm(d["body_lin_vel"][:, body_feet, :2], axis=-1)
        knee_angles = np.rad2deg(d["joint_pos"][:, knees])
        reference_knees = np.rad2deg(d["ref_joint_pos"][:, knees])
        near_upright = (height > 0.85) & (tilt < 20)
        slow_upright = near_upright & (angular_speed < 0.5) & (feet_fz.min(axis=-1) > 20)
        sustained_start = first_sustained_index(slow_upright, max(1, round(0.2 / dt)))
        peak = int(np.argmax(height))
        late = ref_t >= 6.0
        summary = {
            "episode": int(episode), "samples": len(t),
            "frame_first": int(d["frame"][0]), "frame_last": int(d["frame"][-1]),
            "max_height_m": float(height[peak]), "max_height_at_s": float(t[peak]),
            "tilt_at_max_height_deg": float(tilt[peak]),
            "near_upright_duration_s": float(near_upright.sum() * dt),
            "slow_two_foot_upright_duration_s": float(slow_upright.sum() * dt),
            "first_slow_two_foot_upright_for_0p2s_at": None if sustained_start is None else float(t[sustained_start]),
            "near_upright_definition": "height>0.85m and tilt<20deg; descriptive threshold, not a validated success criterion",
            "slow_two_foot_definition": "near_upright and angular_speed<0.5rad/s and each ankle-roll Fz>20N",
            "raw_rewards_after_6s": {
                name: float(d["reward_raw"][late, index].mean())
                for index, name in enumerate(rewards) if np.any(late)
            },
            "checkpoints_in_time": [],
            "knee_extension_timing": {},
            "raw_rewards_after_6p4s": {
                name: float(d["reward_raw"][ref_t >= 6.4, index].mean())
                for index, name in enumerate(rewards) if np.any(ref_t >= 6.4)
            },
        }
        for j, label in enumerate(["left", "right"]):
            reference_crossing = np.flatnonzero((t >= 4) & (reference_knees[:, j] < 90))
            actual_crossing = np.flatnonzero((t >= 4) & (knee_angles[:, j] < 90))
            summary["knee_extension_timing"][label] = {
                "criterion": "first knee angle below 90 degrees after t=4s",
                "reference_time_s": None if not len(reference_crossing) else float(t[reference_crossing[0]]),
                "actual_time_s": None if not len(actual_crossing) else float(t[actual_crossing[0]]),
            }
        for target in [3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]:
            i = int(np.argmin(np.abs(t - target)))
            summary["checkpoints_in_time"].append({
                "t_s": float(t[i]), "ref_height_m": float(d["ref_anchor_pos"][i, 2]),
                "height_m": float(height[i]), "ref_tilt_deg": float(ref_tilt[i]),
                "tilt_deg": float(tilt[i]), "orientation_error_deg": float(error[i]),
                "xy_error_m": float(xy_error[i]),
                "knee_deg_LR": knee_angles[i].tolist(), "ref_knee_deg_LR": reference_knees[i].tolist(),
                "angular_speed_rad_s": float(angular_speed[i]), "ref_angular_speed_rad_s": float(ref_angular_speed[i]),
                "linear_speed_m_s": float(linear_speed[i]), "foot_Fz_N_LR": feet_fz[i].tolist(),
            })
        for i in range(len(t)):
            row = {
                "episode": int(episode), "time_s": t[i], "reference_frame": int(d["frame"][i]),
                "height_m": height[i], "ref_height_m": d["ref_anchor_pos"][i, 2],
                "tilt_deg": tilt[i], "ref_tilt_deg": ref_tilt[i], "ori_error_deg": error[i],
                "xy_error_m": xy_error[i], "angular_speed_rad_s": angular_speed[i],
                "ref_angular_speed_rad_s": ref_angular_speed[i],
                "linear_speed_m_s": linear_speed[i], "ref_linear_speed_m_s": ref_linear_speed[i],
                "knee_L_deg": knee_angles[i, 0], "knee_R_deg": knee_angles[i, 1],
                "ref_knee_L_deg": reference_knees[i, 0], "ref_knee_R_deg": reference_knees[i, 1],
                "foot_L_Fz_N": feet_fz[i, 0], "foot_R_Fz_N": feet_fz[i, 1],
                "foot_L_xy_speed": foot_speed[i, 0], "foot_R_xy_speed": foot_speed[i, 1],
                "done_after_action": bool(d["done_after_action"][i]),
            }
            row.update({"raw_" + name: d["reward_raw"][i, j] for j, name in enumerate(rewards)})
            csv_rows.append(row)

        fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
        axes[0, 0].plot(t, d["ref_anchor_pos"][:, 2], "k--", label="reference")
        axes[0, 0].plot(t, height, label="policy")
        axes[0, 0].set_ylabel("Base height (m)")
        axes[0, 1].plot(t, ref_tilt, "k--", label="reference tilt")
        axes[0, 1].plot(t, tilt, label="policy tilt")
        axes[0, 1].plot(t, error, label="orientation error", alpha=0.7)
        axes[0, 1].set_ylabel("Degrees")
        for j, label in enumerate(["L knee", "R knee"]):
            color = f"C{j}"
            axes[1, 0].plot(t, reference_knees[:, j], "--", color=color, label="ref " + label)
            axes[1, 0].plot(t, knee_angles[:, j], color=color, label=label)
        axes[1, 0].set_ylabel("Knee joint angle (deg)")
        axes[1, 1].plot(t, ref_angular_speed, "k--", label="reference")
        axes[1, 1].plot(t, angular_speed, label="policy")
        axes[1, 1].set_ylabel("Base angular speed (rad/s)")
        for name in ["motion_global_anchor_pos", "motion_global_anchor_ori", "motion_body_pos", "motion_body_ori"]:
            axes[2, 0].plot(t, d["reward_raw"][:, rewards.index(name)], label=name.removeprefix("motion_"))
        axes[2, 0].set_ylabel("Instantaneous unweighted reward")
        for j, label in enumerate(foot_names):
            axes[2, 1].plot(t, feet_fz[:, j], label=label.removeprefix("LINK_"))
        axes[2, 1].set_ylabel("Ankle-roll contact Fz (N)")
        for ax in axes.flat:
            ax.axvspan(4.0, 6.2, color="gray", alpha=0.10)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Episode time (s); shaded: reference lift-to-upright interval")
        fig.suptitle(f"j9519 model_29999 | episode {episode} | pre-action aligned samples")
        fig.tight_layout()
        fig.savefig(directory / f"episode_{episode}.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 5))
        for name in ["motion_global_anchor_pos", "motion_global_anchor_ori", "motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel"]:
            ax.plot(t, d["reward_raw"][:, rewards.index(name)], label=name.removeprefix("motion_"))
        ax.axvspan(4.0, 5.5, color="gray", alpha=0.1, label="lift/fall transition")
        ax.set(xlabel="Episode time (s)", ylabel="Unweighted instantaneous reward", ylim=(-0.02, 1.02),
               title=f"j9519 episode {episode}: pose rewards collapse; velocity rewards recover after lying down")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower left", fontsize=9)
        fig.tight_layout()
        fig.savefig(directory / f"reward_components_episode_{episode}.png", dpi=140)
        plt.close(fig)
        report["episodes"].append(summary)
    with (directory / "tracking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (directory / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory)
