import ast
from pathlib import Path


CONFIG_PATH = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/flat_env_cfg.py")
REGISTRY_PATH = Path("source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/__init__.py")

EXPECTED_BASELINES = {
    "T800FlatBaselineStraightPunchLEnvCfg": ("straight_punch_L", 1.20),
    "T800FlatBaselineStraightPunchREnvCfg": ("straight_punch_R", 1.00),
    "T800FlatBaselineLeftHookEnvCfg": ("left_hook_001", 2.60),
    "T800FlatBaselineRearHookEnvCfg": ("rear_hook_001", 3.54),
    "T800FlatBaselineLeftFrontKickEnvCfg": ("left_front_kick_002", 2.58),
    "T800FlatBaselineRoundhouseKickEnvCfg": ("roundhouse_kick_001", 3.18),
}


def _class_nodes(source: str) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}


def test_six_baseline_tasks_are_registered_with_the_shared_ppo_config():
    registry = REGISTRY_PATH.read_text()
    task_names = (
        "StraightPunchL",
        "StraightPunchR",
        "LeftHook",
        "RearHook",
        "LeftFrontKick",
        "RoundhouseKick",
    )

    for task_name in task_names:
        assert f'id="Tracking-Flat-T800-Baseline-{task_name}-v0"' in registry
    assert registry.count("T800FlatPPORunnerCfg") >= len(task_names)


def test_two_complete_episode_kick_variants_are_registered():
    registry = REGISTRY_PATH.read_text()

    assert 'id="Tracking-Flat-T800-Baseline-LeftFrontKick-CompleteEpisode-v0"' in registry
    assert 'id="Tracking-Flat-T800-Baseline-RoundhouseKick-CompleteEpisode-v0"' in registry


def test_front_kick_height_fall_only_variant_removes_orientation_termination():
    registry = REGISTRY_PATH.read_text()
    config = CONFIG_PATH.read_text()

    assert 'id="Tracking-Flat-T800-Baseline-LeftFrontKick-CompleteEpisode-HeightFallOnly-v0"' in registry
    block = config.split(
        "class T800FlatBaselineLeftFrontKickCompleteEpisodeHeightFallOnlyEnvCfg(", 1
    )[1].split("@configclass", 1)[0]
    assert "T800FlatBaselineLeftFrontKickCompleteEpisodeEnvCfg" in block
    assert "self.terminations.fall_orientation = None" in block
    assert "fall_height" not in block


def test_baseline_envs_only_vary_motion_and_native_duration():
    source = CONFIG_PATH.read_text()
    classes = _class_nodes(source)

    for class_name, expected_args in EXPECTED_BASELINES.items():
        class_node = classes[class_name]
        calls = [node for node in ast.walk(class_node) if isinstance(node, ast.Call)]
        baseline_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "_configure_short_episode_baseline"
        ]
        assert len(baseline_calls) == 1
        actual_args = tuple(ast.literal_eval(arg) for arg in baseline_calls[0].args[1:])
        assert actual_args == expected_args


def test_baseline_contract_matches_j7236_short_episode_mode():
    source = CONFIG_PATH.read_text()
    helper = source.split("def _configure_short_episode_baseline", 1)[1].split("@configclass", 1)[0]

    expected_lines = (
        "motion_cfg.min_traj_duration = None",
        "motion_cfg.bridge_frames = 0",
        'motion_cfg.sampling_mode = "adaptive"',
        "motion_cfg.phase_sampling_windows = []",
        "motion_cfg.pd_stand_reset_ratio = 0.0",
        "motion_cfg.reset_preroll_frames = 0",
        "motion_cfg.play_from_start = True",
        "motion_cfg.resample_at_motion_end = False",
        "env_cfg.terminations.anchor_pos = None",
        "env_cfg.terminations.anchor_ori = None",
        "env_cfg.terminations.ee_body_pos = None",
        'func=mdp.root_height_below_minimum',
        '"minimum_height": 0.4',
        'func=mdp.bad_orientation',
        '"limit_angle": 1.2',
        "env_cfg.events.push_robot.interval_range_s = (5.0, 10.0)",
    )
    for line in expected_lines:
        assert line in helper


def test_training_accepts_motion_file_embedded_in_task_config():
    source = Path("scripts/rsl_rl/train.py").read_text()

    assert "Using motion file configured by task" in source
    assert "select a task with a configured motion file" in source


def test_complete_episode_kicks_add_exactly_one_second_of_terminal_hold():
    source = CONFIG_PATH.read_text()
    classes = _class_nodes(source)
    expected = {
        "T800FlatBaselineLeftFrontKickCompleteEpisodeEnvCfg": (
            "left_front_kick_002_terminal_hold_1s",
            3.58,
        ),
        "T800FlatBaselineRoundhouseKickCompleteEpisodeEnvCfg": (
            "roundhouse_kick_001_terminal_hold_1s",
            4.18,
        ),
    }

    for class_name, expected_args in expected.items():
        calls = [node for node in ast.walk(classes[class_name]) if isinstance(node, ast.Call)]
        baseline_call = next(
            node
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "_configure_short_episode_baseline"
        )
        assert tuple(ast.literal_eval(arg) for arg in baseline_call.args[1:]) == expected_args
        assert any(
            keyword.arg == "motion_dir"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "T800_COMPLETE_EPISODE_MOTION_DIR"
            for keyword in baseline_call.keywords
        )
