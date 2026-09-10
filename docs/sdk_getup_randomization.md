# SDK get-up domain randomization (2026-09-09)

Applies to the SDK supine V1--V4 PPO and prone V3--V4 PPO classes through their
shared bases. Generic GetUp/Smooth, kicks and existing running processes are
unchanged. New launches of these SDK task IDs use this configuration; old
checkpoints played with current task definitions also receive these perturbations.
No jobs were submitted/restarted and no client files were synchronized.

Startup events: all collider contact offsets uniform 0.006--0.014 m (rest
offset unchanged); LINK_BASE mass additive uniform -5--+5 kg using IsaacLab's
standard mass event. Existing friction and COM randomization remain intact.

Every motion reset: reference joint positions scaled independently by 0.8--1.2
then clipped to existing soft limits; reference joint velocities receive additive
-1--+1 rad/s noise. Reference data itself is not modified. Root perturbations
are relative to the reference, in the existing command reset coordinate convention:

| Field | Range |
|---|---|
| x/y position | +/-0.02 m |
| z position | 0--0.03 m |
| roll/pitch | +/-0.03 rad |
| yaw | +/-0.1 rad |
| x/y/z velocity | +/-0.1 m/s |
| roll/pitch/yaw angular velocity | +/-0.2 rad/s |

Root ranges intentionally differ from locomotion reset ranges: do not teleport a
lying reference to standing height. Positive root lift does not guarantee no
penetration after joint/rotation noise; GPU reset/contact smoke is still required.
Rewards, PD, trajectory, observation sizes and pushes are unchanged.

Validation: CPU config/reset tests exercise actual reset function extracted via AST,
verify perturbations are written to simulation inputs without modifying references,
and check identity defaults. These do not validate PhysX contact/mass application.
