from pathlib import Path


def test_540_no_early_terminations_task_is_registered():
    registry = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py").read_text()

    assert 'id="Tracking-Flat-T800-540Huixuanti1-NoEarlyTerminations-v0"' in registry
    assert '"env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1NoEarlyTerminationsEnvCfg' in registry


def test_540_no_early_terminations_cfg_disables_error_terminations_only():
    config = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py").read_text()

    helper = config.split("def _disable_early_terminations(env_cfg):", 1)[1].split("@configclass", 1)[0]
    block = config.split(
        "class T800Flat540Huixuanti1NoEarlyTerminationsEnvCfg(T800Flat540Huixuanti1EnvCfg):",
        1,
    )[1].split("@configclass", 1)[0]

    assert "env_cfg.terminations.anchor_pos = None" in helper
    assert "env_cfg.terminations.anchor_ori = None" in helper
    assert "env_cfg.terminations.ee_body_pos = None" in helper
    assert "time_out" not in helper
    assert "_disable_early_terminations(self)" in block
