"""Late ground-support cost for the 151-frame SDK supine experiments."""

import torch

from .getup_recovery import _motion_phase, smoothstep_phase_ramp


def supine_support_slip(env, command_name, foot_cfg, sensor_names,
                        phase_start=0.78, phase_end=0.84, speed_std=0.20):
    """Bounded ankle-body XY speed proxy, only on ground-bearing feet late in recovery.

    Filtered upward ground forces reject airborne feet and self contact. This
    measures body COM speed, not an exact moving sole contact-point velocity.
    """
    command = env.command_manager.get_term(command_name)
    robot = env.scene[foot_cfg.name]
    velocity = robot.data.body_lin_vel_w[:, foot_cfg.body_ids, :2]
    forces = []
    for name in sensor_names:
        matrix = env.scene[name].data.force_matrix_w
        if matrix is None:
            raise RuntimeError(f"Ground force filter unavailable: {name}")
        forces.append(matrix[..., 2].sum(dim=(1, 2)))
    contact = ((torch.stack(forces, dim=-1) - 10.0) / 40.0).clamp(0.0, 1.0)
    cost = 1.0 - torch.exp(-velocity.square().sum(-1) / speed_std**2)
    ramp = smoothstep_phase_ramp(_motion_phase(command), phase_start, phase_end)
    return (cost * contact).mean(-1) * ramp
