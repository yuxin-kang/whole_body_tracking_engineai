from __future__ import annotations

import argparse
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


PAPER_FASTSAC_STAGE_I_TASK = "Tracking-Flat-G1-1307-PaperFastSAC-Stage-I-v0"
PAPER_FASTSAC_STAGE_II_TASK = "Tracking-Flat-G1-1307-PaperFastSAC-Stage-II-v0"
PAPER_FASTSAC_STAGE_III_TASK = "Tracking-Flat-G1-1307-PaperFastSAC-Stage-III-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate G1 GRSI state files for paper-full recovery training.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=("gravity-release", "schema-smoke"), default="gravity-release")
    parser.add_argument("--task", default=PAPER_FASTSAC_STAGE_I_TASK)
    parser.add_argument("--motion_file", default="data/g1/1307.npz")
    parser.add_argument("--num_states", type=int, default=512)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--joint_count", type=int, default=29)
    parser.add_argument("--source", default="isaaclab-grsi")
    parser.add_argument("--release_steps", type=int, default=32)
    parser.add_argument("--max_attempts", type=int, default=64)
    parser.add_argument("--joint_pos_noise", type=float, default=0.15)
    parser.add_argument("--root_xy_noise", type=float, default=0.05)
    parser.add_argument("--root_z_noise", type=float, default=0.04)
    parser.add_argument("--min_root_height", type=float, default=0.25)
    parser.add_argument("--max_root_height", type=float, default=1.35)
    parser.add_argument("--max_root_speed", type=float, default=8.0)
    parser.add_argument("--max_joint_speed", type=float, default=40.0)
    parser.add_argument("--contact_height_margin", type=float, default=0.035)
    parser.add_argument("--friction_static_min", type=float, default=0.3)
    parser.add_argument("--friction_static_max", type=float, default=1.6)
    parser.add_argument("--friction_dynamic_min", type=float, default=0.3)
    parser.add_argument("--friction_dynamic_max", type=float, default=1.2)
    parser.add_argument("--recombination_fraction", type=float, default=0.5)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from whole_body_tracking.tasks.tracking.config.g1 import g1_mdp  # noqa: E402
from whole_body_tracking.tasks.tracking.config.g1.grsi import (  # noqa: E402
    augment_grsi_with_rotational_recombination,
    make_grsi_state_dict,
    root_state_wxyz_to_xyzw,
    save_grsi_state_file,
    select_diverse_grsi_pool_indexes,
    summarize_support_contacts,
    validate_grsi_production_contract_with_report,
    validate_grsi_state_dict,
)
from whole_body_tracking.tasks.tracking.config.g1.paper_contract import (  # noqa: E402
    G1_PAPER_EQUIVALENCE_CONTRACT_ID,
    G1_PAPER_EQUIVALENCE_MIN_GRSI_STATES,
)
from whole_body_tracking.tasks.tracking.config.g1.paper_full_env_cfg import (  # noqa: E402
    G1Flat1307PaperFastSACStageIEnvCfg,
    G1Flat1307PaperFastSACStageIIEnvCfg,
    G1Flat1307PaperFastSACStageIIIEnvCfg,
)
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1_STANDING_INIT_FILE  # noqa: E402


TASK_CFGS = {
    PAPER_FASTSAC_STAGE_I_TASK: G1Flat1307PaperFastSACStageIEnvCfg,
    PAPER_FASTSAC_STAGE_II_TASK: G1Flat1307PaperFastSACStageIIEnvCfg,
    PAPER_FASTSAC_STAGE_III_TASK: G1Flat1307PaperFastSACStageIIIEnvCfg,
}


def _default_output_path(mode: str) -> Path:
    if mode == "schema-smoke":
        return Path("data/g1/grsi_states_schema_smoke.pth")
    return Path("data/g1/grsi_states.pth")


def _resolve_output_path() -> Path:
    output_path = Path(args_cli.output) if args_cli.output is not None else _default_output_path(args_cli.mode)
    if args_cli.mode == "schema-smoke" and output_path.name == "grsi_states.pth":
        raise ValueError("schema-smoke output cannot overwrite the canonical production pool data/g1/grsi_states.pth")
    return output_path


def _make_schema_smoke_state_dict() -> dict:
    root = torch.zeros(args_cli.num_states, 13)
    root[:, 2] = 0.78
    root[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    root[:, 7:13] = torch.randn(args_cli.num_states, 6) * 0.02
    dof_pos = torch.randn(args_cli.num_states, args_cli.joint_count) * 0.05
    dof_vel = torch.randn(args_cli.num_states, args_cli.joint_count) * 0.05
    joint_names = [f"joint_{idx}" for idx in range(args_cli.joint_count)]
    return make_grsi_state_dict(
        robot_root_states_xyzw=root,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        joint_names=joint_names,
        body_names=["pelvis"],
        source=f"{args_cli.source}:schema-smoke",
        generation_config={
            "mode": "schema_smoke",
            "num_states": args_cli.num_states,
            "joint_count": args_cli.joint_count,
        },
    )


def _make_env() -> ManagerBasedRLEnv:
    if args_cli.task not in TASK_CFGS:
        known = ", ".join(sorted(TASK_CFGS))
        raise ValueError(f"Unsupported GRSI task {args_cli.task!r}. Known tasks: {known}")
    cfg = TASK_CFGS[args_cli.task]()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.commands.motion.motion_file = str(Path(args_cli.motion_file).expanduser().resolve())
    cfg.commands.motion.debug_vis = False
    # GRSI generation must not depend on the canonical recovery pool it is creating.
    cfg.commands.motion.init_pos_file = G1_STANDING_INIT_FILE
    cfg.scene.contact_forces.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    return ManagerBasedRLEnv(cfg=cfg, render_mode=None)


def _sample_initial_states(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    num_envs = int(env_ids.numel())
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] += env.scene.env_origins[env_ids]
    root_state[:, 0:2] += torch.empty(num_envs, 2, device=env.device).uniform_(
        -args_cli.root_xy_noise, args_cli.root_xy_noise
    )
    root_state[:, 2] += torch.empty(num_envs, device=env.device).uniform_(-args_cli.root_z_noise, args_cli.root_z_noise)
    root_state[:, 7:13] = 0.0

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos += torch.empty_like(joint_pos).uniform_(-args_cli.joint_pos_noise, args_cli.joint_pos_noise)
    joint_limits = robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clamp(joint_pos, joint_limits[..., 0], joint_limits[..., 1])
    return root_state, joint_pos, joint_vel


def _write_zero_torque_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    robot = env.scene["robot"]
    zeros = torch.zeros((env_ids.numel(), robot.num_joints), device=env.device)
    robot.write_joint_stiffness_to_sim(0.0, env_ids=env_ids)
    robot.write_joint_damping_to_sim(0.0, env_ids=env_ids)
    robot.set_joint_effort_target(zeros, env_ids=env_ids)
    env.scene.write_data_to_sim()


def _release_once(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    root_state, joint_pos, joint_vel = _sample_initial_states(env, env_ids)
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    _write_zero_torque_state(env, env_ids)

    for _ in range(args_cli.release_steps):
        _write_zero_torque_state(env, env_ids)
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

    return robot.data.root_state_w[env_ids].clone(), robot.data.joint_pos[env_ids].clone(), robot.data.joint_vel[
        env_ids
    ].clone()


def _sample_friction_metadata() -> dict[str, float | str]:
    static_value = float(torch.empty(1).uniform_(args_cli.friction_static_min, args_cli.friction_static_max).item())
    dynamic_value = float(torch.empty(1).uniform_(args_cli.friction_dynamic_min, args_cli.friction_dynamic_max).item())
    return {
        "sampling_scope": "per_release_batch",
        "static_value": static_value,
        "dynamic_value": dynamic_value,
    }


def _accepted_mask(
    root_wxyz: torch.Tensor, dof_pos: torch.Tensor, dof_vel: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    finite = torch.isfinite(root_wxyz).all(dim=1) & torch.isfinite(dof_pos).all(dim=1) & torch.isfinite(dof_vel).all(dim=1)
    quat_norm = torch.linalg.norm(root_wxyz[:, 3:7], dim=1)
    valid_quat = torch.abs(quat_norm - 1.0) < 1.0e-2
    root_height = root_wxyz[:, 2]
    valid_height = (root_height >= args_cli.min_root_height) & (root_height <= args_cli.max_root_height)
    root_speed = torch.linalg.norm(root_wxyz[:, 7:10], dim=1)
    valid_root_speed = root_speed <= args_cli.max_root_speed
    joint_speed = torch.amax(torch.abs(dof_vel), dim=1)
    valid_joint_speed = joint_speed <= args_cli.max_joint_speed
    checks = {
        "non_finite": ~finite,
        "invalid_quaternion": ~valid_quat,
        "root_height": ~valid_height,
        "root_speed": ~valid_root_speed,
        "joint_speed": ~valid_joint_speed,
    }
    return finite & valid_quat & valid_height & valid_root_speed & valid_joint_speed, checks


def _generate_gravity_release_state_dict() -> dict:
    env = _make_env()
    print(f"[GRSI] created env task={args_cli.task} num_envs={env.num_envs}", flush=True)
    accepted_root: list[torch.Tensor] = []
    accepted_dof_pos: list[torch.Tensor] = []
    accepted_dof_vel: list[torch.Tensor] = []
    accepted_body_pos: list[torch.Tensor] = []
    candidate_rejected = 0
    attempts = 0
    friction_batches: list[dict[str, float | str]] = []
    candidate_rejection_reasons = {
        "non_finite": 0,
        "invalid_quaternion": 0,
        "root_height": 0,
        "root_speed": 0,
        "joint_speed": 0,
    }
    try:
        env.reset()
        print("[GRSI] environment reset complete", flush=True)
        env_ids = torch.arange(env.num_envs, device=env.device)
        robot = env.scene["robot"]
        motion_body_ids = robot.find_bodies(list(g1_mdp.G1_MOTION_BODY_NAMES), preserve_order=True)[0]
        while sum(t.shape[0] for t in accepted_root) < args_cli.num_states and attempts < args_cli.max_attempts:
            attempts += 1
            friction_batches.append(_sample_friction_metadata())
            root_wxyz, dof_pos, dof_vel = _release_once(env, env_ids)
            body_pos_w = robot.data.body_pos_w[env_ids][:, motion_body_ids].clone()
            root_local_wxyz = root_wxyz.clone()
            root_local_wxyz[:, 0:3] -= env.scene.env_origins[env_ids]
            mask, check_failures = _accepted_mask(root_local_wxyz, dof_pos, dof_vel)
            candidate_rejected += int((~mask).sum().item())
            for key, failure_mask in check_failures.items():
                candidate_rejection_reasons[key] += int(failure_mask.sum().item())
            if mask.any():
                accepted_root.append(root_state_wxyz_to_xyzw(root_local_wxyz[mask]).detach().cpu())
                accepted_dof_pos.append(dof_pos[mask].detach().cpu())
                accepted_dof_vel.append(dof_vel[mask].detach().cpu())
                accepted_body_pos.append(body_pos_w[mask].detach().cpu())
            current_accepted = sum(t.shape[0] for t in accepted_root)
            print(
                f"[GRSI] attempt={attempts}/{args_cli.max_attempts} accepted={current_accepted}/{args_cli.num_states} "
                f"candidate_rejected={candidate_rejected}",
                flush=True,
            )

        accepted = sum(t.shape[0] for t in accepted_root)
        if accepted == 0:
            raise RuntimeError("GRSI gravity-release generated zero accepted states; relax thresholds or inspect robot setup.")
        if accepted < args_cli.num_states:
            raise RuntimeError(
                f"GRSI gravity-release accepted only {accepted}/{args_cli.num_states} states after "
                f"{attempts} attempts; increase --max_attempts or relax thresholds."
            )

        root_xyzw = torch.cat(accepted_root, dim=0)[: args_cli.num_states]
        dof_pos = torch.cat(accepted_dof_pos, dim=0)[: args_cli.num_states]
        dof_vel = torch.cat(accepted_dof_vel, dim=0)[: args_cli.num_states]
        body_pos_w = torch.cat(accepted_body_pos, dim=0)[: args_cli.num_states]
        rotational_recombination = {
            "enabled": False,
            "augmented_count": 0,
            "rotation_source": "disabled",
        }
        if root_xyzw.shape[0] > 1 and args_cli.recombination_fraction > 0.0:
            num_augmented = min(
                max(int(args_cli.num_states * args_cli.recombination_fraction), 1),
                int(root_xyzw.shape[0]),
            )
            base_indexes = torch.arange(num_augmented, dtype=torch.long)
            donor_indexes = torch.roll(base_indexes, shifts=1)
            aug_root, aug_dof_pos, aug_dof_vel, rotational_recombination = augment_grsi_with_rotational_recombination(
                root_xyzw,
                dof_pos,
                dof_vel,
                base_indexes=base_indexes,
                donor_indexes=donor_indexes,
            )
            combined_root = torch.cat([root_xyzw, aug_root], dim=0)
            combined_dof_pos = torch.cat([dof_pos, aug_dof_pos], dim=0)
            combined_dof_vel = torch.cat([dof_vel, aug_dof_vel], dim=0)
            combined_body_pos_w = torch.cat([body_pos_w, body_pos_w[base_indexes]], dim=0)
            selected_indexes = select_diverse_grsi_pool_indexes(combined_root, combined_dof_pos, args_cli.num_states).cpu()
            root_xyzw = combined_root[selected_indexes]
            dof_pos = combined_dof_pos[selected_indexes]
            dof_vel = combined_dof_vel[selected_indexes]
            body_pos_w = combined_body_pos_w[selected_indexes]
        accepted = int(root_xyzw.shape[0])
        friction_summary = {
            "sampling_scope": "per_release_batch",
            "batch_count": len(friction_batches),
            "static_range": [args_cli.friction_static_min, args_cli.friction_static_max],
            "dynamic_range": [args_cli.friction_dynamic_min, args_cli.friction_dynamic_max],
            "static_samples": [float(batch["static_value"]) for batch in friction_batches[:16]],
            "dynamic_samples": [float(batch["dynamic_value"]) for batch in friction_batches[:16]],
        }
        return make_grsi_state_dict(
            robot_root_states_xyzw=root_xyzw,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            joint_names=list(g1_mdp.G1_MOTION_JOINT_NAMES),
            body_names=list(g1_mdp.G1_MOTION_BODY_NAMES),
            source=f"{args_cli.source}:gravity-release",
            generation_config={
                "mode": "gravity_release",
                "task": args_cli.task,
                "motion_file": str(Path(args_cli.motion_file).expanduser().resolve()),
                "requested_num_states": args_cli.num_states,
                "minimum_required_states": G1_PAPER_EQUIVALENCE_MIN_GRSI_STATES,
                "num_envs": args_cli.num_envs,
                "release_steps": args_cli.release_steps,
                "attempts": attempts,
                "accepted": accepted,
                "candidate_rejected": candidate_rejected,
                "candidate_rejection_reasons": candidate_rejection_reasons,
                "zero_torque": True,
                "friction_randomization": friction_summary,
                "rotational_recombination": rotational_recombination,
                "source_contract_id": G1_PAPER_EQUIVALENCE_CONTRACT_ID,
                "joint_pos_noise": args_cli.joint_pos_noise,
                "root_xy_noise": args_cli.root_xy_noise,
                "root_z_noise": args_cli.root_z_noise,
                "min_root_height": args_cli.min_root_height,
                "max_root_height": args_cli.max_root_height,
                "max_root_speed": args_cli.max_root_speed,
                "max_joint_speed": args_cli.max_joint_speed,
            },
            contact_summary=summarize_support_contacts(
                body_pos_w=body_pos_w,
                body_names=list(g1_mdp.G1_MOTION_BODY_NAMES),
                contact_height_margin=args_cli.contact_height_margin,
            ),
        )
    finally:
        env.close()


def main() -> None:
    output_path = _resolve_output_path()
    if args_cli.mode == "schema-smoke":
        data = _make_schema_smoke_state_dict()
        expected_joint_names = [f"joint_{idx}" for idx in range(args_cli.joint_count)]
    else:
        data = _generate_gravity_release_state_dict()
        expected_joint_names = list(g1_mdp.G1_MOTION_JOINT_NAMES)

    report = validate_grsi_state_dict(data, expected_joint_names=expected_joint_names)
    if output_path.name == "grsi_states.pth":
        validate_grsi_production_contract_with_report(data, output_path, report)
    save_grsi_state_file(output_path, data)
    print(
        f"[GRSI] wrote {output_path} mode={data['generation_config']['mode']} "
        f"accepted={report.accepted} rejected={report.rejected} schema_version={data['version']}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
