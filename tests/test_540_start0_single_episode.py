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


def test_t800_base_uses_adaptive_random_phase_without_motion_bridging():
    config = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py").read_text()
    block = config.split("class T800FlatEnvCfg(TrackingEnvCfg):", 1)[1].split("@configclass", 1)[0]

    assert 'self.commands.motion.sampling_mode = "adaptive"' in block
    assert "self.commands.motion.play_from_start = False" in block
    assert "self.commands.motion.resample_at_motion_end = True" in block
    assert "self.commands.motion.min_traj_duration = None" in block
    assert "self.commands.motion.bridge_frames = 0" in block


def test_native_motion_helper_preserves_task_phase_and_end_behavior():
    source = Path("scripts/rsl_rl/train.py").read_text()
    block = source.split("def _configure_t800_motion_episode", 1)[1].split(
        "def _configure_t800_punch_normal_episode", 1
    )[0]

    assert "motion_cfg.play_from_start =" not in block
    assert "motion_cfg.resample_at_motion_end =" not in block
    assert "phase_mode =" in block
    assert "end_mode =" in block


def test_punch_normal_episode_mode_restores_successful_training_distribution():
    source = Path("scripts/rsl_rl/train.py").read_text()
    block = source.split("def _configure_t800_punch_normal_episode", 1)[1].split(
        "def _learn_with_code_state_fallback", 1
    )[0]

    assert "env_cfg.episode_length_s = 10.0" in block
    assert "motion_cfg.min_traj_duration = 10.0" in block
    assert "motion_cfg.bridge_frames = 20" in block
    assert 'motion_cfg.sampling_mode = "adaptive"' in block
    assert "motion_cfg.pd_stand_reset_ratio = 0.2" in block
    assert "motion_cfg.play_from_start = False" in block
    assert "motion_cfg.resample_at_motion_end = True" in block
    assert "env_cfg.events.push_robot.interval_range_s = (1.0, 3.0)" in block
    assert 'choices=("native", "punch_normal")' in source
    assert 'args_cli.t800_episode_mode == "punch_normal"' in source


def test_punch_normal_episode_mode_uses_original_randomized_runner_episode_lengths():
    source = Path("scripts/rsl_rl/train.py").read_text()

    assert 'is_native_t800_episode = is_t800_task and args_cli.t800_episode_mode == "native"' in source
    assert "init_at_random_ep_len=not is_native_t800_episode" in source


def test_t800_playback_uses_native_episode_and_holds_at_motion_end():
    play_source = Path("scripts/rsl_rl/play.py").read_text()

    assert "env_cfg.episode_length_s = num_frames / fps" in play_source
    assert "motion_cfg.play_from_start = True" in play_source
    assert "motion_cfg.resample_at_motion_end = False" in play_source
    assert "_configure_t800_motion_episode(" in play_source
