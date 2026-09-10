import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


path = Path("source/whole_body_tracking/whole_body_tracking/utils/getup_scratch.py")
spec = importlib.util.spec_from_file_location("getup_scratch_test_helper", path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def fresh_runner():
    from rsl_rl.modules.normalizer import EmpiricalNormalization
    policy = torch.nn.Linear(3, 2)
    return SimpleNamespace(
        current_learning_iteration=0,
        alg=SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters(), lr=.001)),
        obs_normalizer=EmpiricalNormalization([3]), privileged_obs_normalizer=EmpiricalNormalization([4]),
    )


def test_initialization_proves_fresh_model_optimizer_and_normalizers():
    result = helper.audit_scratch_initialization(fresh_runner())
    assert result["first_iteration"] == 0
    assert result["optimizer_state_count"] == 0
    assert result["normalization_counts"] == [0, 0]
    assert result["initial_optimizer_lrs"] == [.001]
    assert len(result["initial_model_sha256"]) == 64


@pytest.mark.parametrize("contamination", ["iteration", "normalizer", "optimizer", "nan"])
def test_resume_contamination_is_rejected(contamination):
    runner = fresh_runner()
    if contamination == "iteration":
        runner.current_learning_iteration = 30000
    elif contamination == "normalizer":
        runner.obs_normalizer(torch.ones(2, 3))
    elif contamination == "optimizer":
        runner.alg.policy(torch.ones(2, 3)).sum().backward()
        runner.alg.optimizer.step()
    else:
        with torch.no_grad():
            runner.alg.policy.weight.fill_(float("nan"))
    with pytest.raises(RuntimeError):
        helper.audit_scratch_initialization(runner)


def samples():
    return {"frame": np.tile(np.arange(379)[:, None], (1, 3)),
            **{key: np.zeros((379, 3)) for key in ("tilt", "angular_speed", "orientation_error", "position_error")},
            "height": np.ones((379, 3))}


def test_height_peak_or_partial_initialization_not_full_start_pass():
    data = samples()
    data["height"][350, 1] = .1
    data["frame"][:, 2] = np.clip(data["frame"][:, 2] + 200, 0, 378)
    summary = helper.summarize_full_start(data, .02)
    assert summary["posture_pass_per_env"] == [True, False, False]
    assert summary["maximum_height_per_env"] == [1., 1., 1.]
    assert "NOT verified bilateral" in summary["note"]


def test_from_scratch_launcher_and_entry_contract():
    launcher = Path("scripts/slurm/train_t800_getup_4x3090_aggressive.sbatch").read_text()
    assert "#SBATCH --gres=gpu:rtx3090:4" in launcher
    assert "--gres=gpu:rtx3090:1" in launcher
    assert "for variant in a b c d" in launcher
    assert launcher.index("run_four preflight") < launcher.index("run_four train")
    assert '"preflight-only"' in launcher
    assert "export HOME=" not in launcher
    entry = Path("scripts/rsl_rl/train_getup_aggressive.py").read_text()
    assert "agent.resume = False" in entry
    assert "runner.load(" not in entry and "torch.load(" not in entry
    assert "audit_scratch_initialization(runner)" in entry
    assert "runner.current_learning_iteration = completed" in entry
    assert "evaluate_full_start(" in entry


def test_evaluation_restores_training_phase_and_normalization(tmp_path):
    runner = fresh_runner()
    command = SimpleNamespace(
        _play_from_start=False, motion=SimpleNamespace(time_step_total=379),
        time_steps=torch.tensor([200, 200]),
        robot_anchor_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
        anchor_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
        robot_anchor_pos_w=torch.tensor([[0., 0., 1.]]).repeat(2, 1),
        anchor_pos_w=torch.tensor([[0., 0., 1.]]).repeat(2, 1),
        robot_anchor_ang_vel_w=torch.zeros(2, 3), robot_joint_pos=torch.zeros(2, 2),
        joint_pos=torch.zeros(2, 2), cfg=SimpleNamespace(motion_joint_names=["left", "right"]),
    )
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(2, 2, 3)), body_names=["left", "right"])
    base = SimpleNamespace(
        num_envs=2, common_step_counter=480000, _sim_step_counter=1920000, step_dt=.02,
        command_manager=SimpleNamespace(get_term=lambda _: command),
        scene=SimpleNamespace(sensors={"contact_forces": sensor}),
    )

    class Env:
        unwrapped = base

        def reset(self):
            torch.rand(3)
            np.random.rand(3)
            random.random()
            command.time_steps.fill_(0 if command._play_from_start else 200)
            return torch.zeros(2, 3), {}

        def step(self, actions):
            base.common_step_counter += 1
            base._sim_step_counter += 4
            command.time_steps.add_(1).clamp_(max=378)
            return torch.zeros(2, 3), None, None, None

    runner.env = Env()
    runner.device = "cpu"

    def policy_getter(device):
        runner.obs_normalizer.eval()
        return lambda obs: runner.alg.policy(runner.obs_normalizer(obs))

    runner.get_inference_policy = policy_getter
    runner.train_mode = lambda: runner.obs_normalizer.train()
    before_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    result = helper.evaluate_full_start(runner, tmp_path / "eval", 4999, num_reported_envs=1)
    assert result["posture_pass_per_env"] == [True]
    assert result["num_envs_evaluated"] == 1
    assert command._play_from_start is False
    assert command.time_steps.tolist() == [200, 200]
    assert base.common_step_counter == 480000
    assert base._sim_step_counter == 1920000
    assert int(runner.obs_normalizer.count) == 0
    assert runner.obs_normalizer.training
    assert torch.equal(before_rng[0], torch.get_rng_state())
    np.testing.assert_array_equal(before_rng[1][1], np.random.get_state()[1])
    assert before_rng[1][2:] == np.random.get_state()[2:]
    assert before_rng[2] == random.getstate()
    assert (tmp_path / "eval/rollout.npz").is_file()


def test_learn_chunk_normalizes_only_first_observation_without_updating_stats():
    runner = fresh_runner()
    runner.current_learning_iteration = 8
    runner.obs_normalizer(torch.ones(2, 3))
    runner.privileged_obs_normalizer(torch.ones(2, 4) * 2)
    obs = torch.ones(2, 3) * 5
    extras = {"observations": {"critic": torch.ones(2, 4) * 7}}
    reader = lambda: (obs, extras)
    runner.env = SimpleNamespace(get_observations=reader)
    seen = []

    def learn(**kwargs):
        assert kwargs == {"num_learning_iterations": 8, "init_at_random_ep_len": False}
        seen.append(runner.env.get_observations())
        seen.append(runner.env.get_observations())

    runner.learn = learn
    helper.learn_chunk(runner, 8)
    assert torch.allclose(seen[0][0], torch.ones(2, 3) * 400)
    assert torch.allclose(seen[0][1]["observations"]["critic"], torch.ones(2, 4) * 500)
    assert seen[1][0] is obs
    assert seen[1][1] is extras
    assert runner.env.get_observations is reader
    assert int(runner.obs_normalizer.count) == 2
    assert int(runner.privileged_obs_normalizer.count) == 2
    assert runner.obs_normalizer.training


def test_learn_chunk_restores_reader_on_exception():
    runner = fresh_runner()
    runner.current_learning_iteration = 8
    reader = lambda: (None, None)
    runner.env = SimpleNamespace(get_observations=reader)

    def fail(**kwargs):
        raise RuntimeError("fake upstream failure")

    runner.learn = fail
    with pytest.raises(RuntimeError, match="fake upstream failure"):
        helper.learn_chunk(runner, 8)
    assert runner.env.get_observations is reader


def test_preflight_requires_valid_procedure_but_not_a_trained_policy():
    summary = {"uninterrupted_from_frame_zero": [True] * 16, "posture_pass_fraction": 0.0}
    diagnostics = {"reset_source_counts": [12, 10, 10], "physical_source_steps": [10, 20, 30],
                   "peak_contact_force_n": 100., "sampled_contact_impulse_proxy_ns": 20.,
                   "max_reference_joint_limit_violation_rad": 0.01}
    helper.assert_preflight_result(summary, diagnostics, "d", 60)
    summary["uninterrupted_from_frame_zero"][3] = False
    with pytest.raises(RuntimeError, match="reset or skipped"):
        helper.assert_preflight_result(summary, diagnostics, "d", 60)
    summary["uninterrupted_from_frame_zero"][3] = True
    diagnostics["reset_source_counts"][2] = 0
    with pytest.raises(RuntimeError, match="reset sources"):
        helper.assert_preflight_result(summary, diagnostics, "d", 60)
    diagnostics["reset_source_counts"][2] = 10
    with pytest.raises(RuntimeError, match="step accounting"):
        helper.assert_preflight_result(summary, diagnostics, "d", 61)
    diagnostics["peak_contact_force_n"] = float("nan")
    with pytest.raises(RuntimeError, match="Non-finite"):
        helper.assert_preflight_result(summary, diagnostics, "d", 60)
