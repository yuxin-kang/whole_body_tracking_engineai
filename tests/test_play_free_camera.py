import ast
from pathlib import Path


PLAY_PATH = Path("scripts/rsl_rl/play.py")


def test_play_defaults_to_static_nearby_viewer():
    source = PLAY_PATH.read_text()
    tree = ast.parse(source)
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_configure_play_viewer"
    )
    helper_source = ast.get_source_segment(source, helper)

    assert 'env_cfg.viewer.origin_type = "world"' in helper_source
    assert "env_cfg.viewer.eye = (-1.8, -1.8, 1.6)" in helper_source
    assert "env_cfg.viewer.lookat = (0.0, 0.0, 0.8)" in helper_source
    assert "if follow_camera:" in helper_source


def test_follow_camera_is_an_opt_in_cli_flag_applied_before_env_creation():
    source = PLAY_PATH.read_text()

    flag_position = source.index('"--follow_camera"')
    default_position = source.index("default=False", flag_position)
    configure_position = source.index("_configure_play_viewer(env_cfg, follow_camera=args_cli.follow_camera)")
    create_position = source.index("env = gym.make", configure_position)

    assert flag_position < default_position < configure_position < create_position
