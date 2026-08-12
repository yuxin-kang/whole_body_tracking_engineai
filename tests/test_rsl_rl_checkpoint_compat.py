from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path(
    "source/whole_body_tracking/whole_body_tracking/utils/rsl_rl_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("rsl_rl_checkpoint", MODULE_PATH)
checkpoint_compat = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(checkpoint_compat)


class FakeNormalizer(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(size))
        self.register_buffer("_var", torch.ones(size))
        self.register_buffer("_std", torch.ones(size))
        self.register_buffer("count", torch.tensor(1.0))


class EmbeddedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(3, 2)
        self.critic = torch.nn.Linear(4, 1)
        self.actor_obs_normalizer = FakeNormalizer(3)
        self.critic_obs_normalizer = FakeNormalizer(4)


class RunnerPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(3, 2)
        self.critic = torch.nn.Linear(4, 1)


def _runner(policy, *, runner_normalizers: bool):
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=policy),
        device="cpu",
        current_learning_iteration=0,
    )
    if runner_normalizers:
        runner.obs_normalizer = FakeNormalizer(3)
        runner.privileged_obs_normalizer = FakeNormalizer(4)
    return runner


def _normalizer_state(size: int, value: float):
    normalizer = FakeNormalizer(size)
    normalizer._mean.fill_(value)
    normalizer._var.fill_(value + 1.0)
    normalizer._std.fill_(value + 2.0)
    normalizer.count.fill_(123.0)
    return normalizer.state_dict()


def test_loads_top_level_normalizers_into_new_policy_format(tmp_path):
    source = RunnerPolicy()
    path = tmp_path / "rsl2.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "obs_norm_state_dict": _normalizer_state(3, 2.0),
            "privileged_obs_norm_state_dict": _normalizer_state(4, 4.0),
            "iter": 23000,
            "infos": {},
        },
        path,
    )
    target = EmbeddedPolicy()
    runner = _runner(target, runner_normalizers=False)

    checkpoint_compat.load_rsl_rl_checkpoint_for_inference(runner, path)

    assert torch.equal(target.actor.weight, source.actor.weight)
    assert torch.all(target.actor_obs_normalizer._mean == 2.0)
    assert torch.all(target.critic_obs_normalizer._mean == 4.0)
    assert runner.current_learning_iteration == 23000


def test_loads_embedded_normalizers_into_old_runner_format(tmp_path):
    source = EmbeddedPolicy()
    source.actor_obs_normalizer.load_state_dict(_normalizer_state(3, 3.0))
    source.critic_obs_normalizer.load_state_dict(_normalizer_state(4, 5.0))
    path = tmp_path / "rsl3.pt"
    torch.save({"model_state_dict": source.state_dict(), "iter": 29999, "infos": {}}, path)
    target = RunnerPolicy()
    runner = _runner(target, runner_normalizers=True)

    checkpoint_compat.load_rsl_rl_checkpoint_for_inference(runner, path)

    assert torch.equal(target.actor.weight, source.actor.weight)
    assert torch.all(runner.obs_normalizer._mean == 3.0)
    assert torch.all(runner.privileged_obs_normalizer._mean == 5.0)


def test_config_field_detection_uses_annotations_instead_of_hasattr():
    for path in (
        Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/agents/rsl_rl_ppo_cfg.py"),
        Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py"),
    ):
        source = path.read_text()
        assert 'getattr(config_type, "__annotations__", {})' in source
        assert "hasattr(RslRlPpoActorCriticCfg, field_name)" not in source
