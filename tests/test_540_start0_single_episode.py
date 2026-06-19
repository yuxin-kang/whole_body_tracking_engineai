from pathlib import Path


def test_540_start0_task_is_registered():
    registry = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py").read_text()

    assert 'id="Tracking-Flat-T800-540Huixuanti1-Start0-v0"' in registry
    assert '"env_cfg_entry_point": flat_env_cfg.T800Flat540Huixuanti1Start0EnvCfg' in registry


def test_540_start0_cfg_forces_clip_start_without_pd_stand():
    config = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py").read_text()

    block = config.split("class T800Flat540Huixuanti1Start0EnvCfg(T800Flat540Huixuanti1EnvCfg):", 1)[1].split(
        "@configclass", 1
    )[0]

    assert "self.commands.motion.play_from_start = True" in block
    assert "self.commands.motion.resample_at_motion_end = False" in block
    assert "self.commands.motion.pd_stand_reset_ratio = 0.0" in block


def test_motion_command_supports_no_resample_at_motion_end():
    command_source = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py").read_text()

    assert "play_from_start: bool = False" in command_source
    assert "resample_at_motion_end: bool = True" in command_source
    assert "self._play_from_start = bool(self.cfg.play_from_start)" in command_source
    assert "if self.cfg.resample_at_motion_end:" in command_source
    assert "self.time_steps[env_ids] = self.motion.time_step_total - 1" in command_source
