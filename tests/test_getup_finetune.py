import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE = Path("source/whole_body_tracking/whole_body_tracking/utils/getup_finetune.py")
spec = importlib.util.spec_from_file_location("getup_finetune", MODULE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class Runner:
    def __init__(self):
        policy = torch.nn.Linear(3, 2)
        self.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters(), lr=.001),
                                   learning_rate=.001, schedule="adaptive")
        self.obs_normalizer = torch.nn.Linear(3, 3)
        self.privileged_obs_normalizer = torch.nn.Linear(4, 4)
        self.current_learning_iteration = 0

    def load(self, path, load_optimizer):
        assert load_optimizer
        state = torch.load(path, weights_only=False)
        self.alg.policy.load_state_dict(state["model_state_dict"])
        self.obs_normalizer.load_state_dict(state["obs_norm_state_dict"])
        self.privileged_obs_normalizer.load_state_dict(state["privileged_obs_norm_state_dict"])
        self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.current_learning_iteration = state["iter"]


def make_checkpoint(path):
    runner = Runner()
    runner.alg.policy(torch.ones(2, 3)).sum().backward()
    runner.alg.optimizer.step()  # Non-empty Adam moments must survive loading.
    state = {
        "model_state_dict": runner.alg.policy.state_dict(),
        "obs_norm_state_dict": runner.obs_normalizer.state_dict(),
        "privileged_obs_norm_state_dict": runner.privileged_obs_normalizer.state_dict(),
        "optimizer_state_dict": runner.alg.optimizer.state_dict(), "iter": 29999,
    }
    torch.save(state, path)
    return state


def test_resume_restores_all_state_then_overrides_optimizer_lr(tmp_path):
    path = tmp_path / "source.pt"
    state = make_checkpoint(path)
    runner = Runner()
    result = helper.restore_finetune_checkpoint(runner, path, 1e-5)
    assert result["first_iteration"] == 30000
    assert result["source_optimizer_lrs"] == [.001]
    assert runner.alg.schedule == "fixed"
    assert runner.alg.learning_rate == 1e-5
    assert [g["lr"] for g in runner.alg.optimizer.param_groups] == [1e-5]
    helper._assert_state_equal(state["optimizer_state_dict"]["state"], runner.alg.optimizer.state_dict()["state"], "moments")
    assert torch.equal(torch.load(path, weights_only=False)["model_state_dict"]["weight"], state["model_state_dict"]["weight"])


@pytest.mark.parametrize("lr", [0, -1, float("nan"), float("inf")])
def test_invalid_learning_rate_rejected(lr):
    with pytest.raises(ValueError):
        helper.restore_finetune_checkpoint(Runner(), "not-read.pt", lr)


def test_missing_normalizer_fails_closed(tmp_path):
    path = tmp_path / "incomplete.pt"
    state = make_checkpoint(path)
    del state["privileged_obs_norm_state_dict"]
    torch.save(state, path)
    with pytest.raises(ValueError, match="Incomplete"):
        helper.restore_finetune_checkpoint(Runner(), path, 1e-5)


def test_config_delta_guard_rejects_unrelated_changes():
    before = {"rewards": {"motion_body_lin_vel": {"weight": 1.0}}, "physics": 1}
    after = {"rewards": {"motion_body_lin_vel": {"weight": 1.0}, "getup_transition_anchor_ori": {}}, "physics": 1}
    changes = helper.config_changes(before, after)
    helper.assert_reward_only_changes(changes, "ori")
    after["physics"] = 2
    with pytest.raises(ValueError, match="physics"):
        helper.assert_reward_only_changes(helper.config_changes(before, after), "ori")


def test_launcher_uses_two_separate_gpus_and_preflight_barrier():
    source = Path("scripts/slurm/finetune_t800_getup_2x4090.sbatch").read_text()
    assert "#SBATCH --gres=gpu:rtx4090d:2" in source
    assert source.count("--gres=gpu:rtx4090d:1") == 2
    assert "run_pair preflight\nrun_pair train" in source
    assert "export HOME=" not in source
    assert "--iterations 5000" in source


def test_real_runner_can_save_initial_snapshot_before_learn(tmp_path):
    runner_module = Path("source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py")
    runner_spec = importlib.util.spec_from_file_location("local_motion_runner", runner_module)
    runner_module = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner_module)
    env = SimpleNamespace(
        get_observations=lambda: (torch.zeros(2, 3), {"observations": {"critic": torch.zeros(2, 4)}}),
        num_actions=2, num_envs=2,
    )
    cfg = {
        "policy": {"class_name": "ActorCritic", "actor_hidden_dims": [8], "critic_hidden_dims": [8]},
        "algorithm": {"class_name": "PPO", "learning_rate": 1e-5, "schedule": "fixed"},
        "num_steps_per_env": 2, "save_interval": 250, "empirical_normalization": True,
    }
    runner = runner_module.MotionOnPolicyRunner(env, cfg, log_dir=str(tmp_path), device="cpu")
    runner.logger_type = "wandb"
    path = tmp_path / "model_init.pt"
    runner.save(str(path))
    assert path.is_file()
    assert runner.writer is None  # Must not require W&B init for this local save.
    source = Path("scripts/rsl_rl/finetune_getup_recovery.py").read_text()
    assert source.index("runner.logger_type = agent.logger") < source.index('runner.save(str(output / "model_init.pt"))')
