from __future__ import annotations

import ast
import importlib.util
import re
import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree


REPO_ROOT = Path(__file__).resolve().parents[1]
G1_MDP_PATH = (
    REPO_ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/g1_mdp.py"
)
G1_FLAT_ENV_CFG = (
    REPO_ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py"
)
G1_INIT = REPO_ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py"
G1_AGENT_CFG = (
    REPO_ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py"
)
COMMANDS_PATH = REPO_ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
REWARDS_PATH = REPO_ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py"
TERMINATIONS_PATH = REPO_ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py"
REPLAY_PATH = REPO_ROOT / "scripts/replay_npz.py"
G1_ROBOT_PATH = REPO_ROOT / "source/whole_body_tracking/whole_body_tracking/robots/g1.py"
MOTION_PATH = REPO_ROOT / "data/g1/1307.npz"
INIT_STATE_PATH = REPO_ROOT / "data/g1/robot_init_states_8192.pth"
G1_XML_PATH = REPO_ROOT / "source/whole_body_tracking/whole_body_tracking/assets/g1/xmls/g1.xml"


def _load_g1_mdp():
    spec = importlib.util.spec_from_file_location("g1_mdp_under_test", G1_MDP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _npy_shape(payload: bytes) -> tuple[int, ...]:
    assert payload[:6] == b"\x93NUMPY"
    major = payload[6]
    if major == 1:
        header_len = struct.unpack("<H", payload[8:10])[0]
        header_start = 10
    else:
        header_len = struct.unpack("<I", payload[8:12])[0]
        header_start = 12
    header = payload[header_start : header_start + header_len].decode("latin1")
    return tuple(ast.literal_eval(header)["shape"])


def _mjcf_body_names() -> list[str]:
    root = ElementTree.parse(G1_XML_PATH).getroot()
    worldbody = root.find("worldbody")
    assert worldbody is not None
    names: list[str] = []

    def walk(body):
        names.append(body.attrib["name"])
        for child in body.findall("body"):
            walk(child)

    for body in worldbody.findall("body"):
        walk(body)
    return names


def _mjcf_joint_names() -> list[str]:
    root = ElementTree.parse(G1_XML_PATH).getroot()
    worldbody = root.find("worldbody")
    assert worldbody is not None
    names: list[str] = []

    def walk(body):
        for joint in body.findall("joint"):
            names.append(joint.attrib["name"])
        for child in body.findall("body"):
            walk(child)

    for body in worldbody.findall("body"):
        walk(body)
    return names


def test_g1_1307_npz_shapes_match_upstream_motion_contract():
    assert MOTION_PATH.is_file()
    assert INIT_STATE_PATH.is_file()
    assert INIT_STATE_PATH.stat().st_size > 1_000_000
    with zipfile.ZipFile(MOTION_PATH) as archive:
        shapes = {name.removesuffix(".npy"): _npy_shape(archive.read(name)) for name in archive.namelist()}

    assert shapes["fps"] == (1,)
    assert shapes["joint_pos"] == (15793, 29)
    assert shapes["joint_vel"] == (15793, 29)
    assert shapes["body_pos_w"] == (15793, 30, 3)
    assert shapes["body_quat_w"] == (15793, 30, 4)
    assert shapes["body_lin_vel_w"] == (15793, 30, 3)
    assert shapes["body_ang_vel_w"] == (15793, 30, 3)


def test_g1_motion_order_constants_match_the_copied_mjcf():
    g1_mdp = _load_g1_mdp()

    assert g1_mdp.G1_MOTION_JOINT_NAMES == _mjcf_joint_names()
    assert g1_mdp.G1_MOTION_BODY_NAMES == _mjcf_body_names()
    assert len(g1_mdp.G1_TRACKING_BODY_NAMES) == 14
    assert g1_mdp.G1_ANCHOR_BODY_NAME == "torso_link"
    assert g1_mdp.G1_ROOT_BODY_NAME == "pelvis"
    for body_name in g1_mdp.G1_TRACKING_BODY_NAMES:
        assert body_name in g1_mdp.G1_MOTION_BODY_NAMES


def test_g1_standing_reset_pure_helper_preserves_motion_xy_and_converts_quaternion():
    g1_mdp = _load_g1_mdp()
    motion_root = [1.0, 2.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    sampled_root_xyzw = [9.0, 8.0, 0.9, 0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    motion_joint = [0.0, 0.1]
    sampled_joint = [0.2, 0.3]

    root, joint = g1_mdp.apply_standing_reset_branch(
        motion_root,
        sampled_root_xyzw,
        motion_joint,
        sampled_joint,
        use_standing=True,
    )

    assert root[:2] == motion_root[:2]
    assert root[2] == sampled_root_xyzw[2]
    assert root[3:7] == [0.4, 0.1, 0.2, 0.3]
    assert root[7:13] == sampled_root_xyzw[7:13]
    assert joint == sampled_joint


def test_g1_tasks_register_stage_curriculum_and_upstream_ppo_values():
    init_source = G1_INIT.read_text()
    flat_source = G1_FLAT_ENV_CFG.read_text()
    agent_source = G1_AGENT_CFG.read_text()

    for task_id in [
        "Tracking-Flat-G1-v0",
        "Tracking-Flat-G1-Wo-State-Estimation-v0",
        "Tracking-Flat-G1-Standing-v0",
        "Tracking-Flat-G1-1307-Stage-I-v0",
        "Tracking-Flat-G1-1307-Stage-II-v0",
        "Tracking-Flat-G1-1307-Stage-III-v0",
        "Tracking-Flat-G1-1307-Checkpoint-v0",
    ]:
        assert task_id in init_source

    assert "G1_1307_MOTION_FILE" in flat_source
    assert "data/g1/1307.npz" in flat_source
    assert "data/g1/robot_init_states_8192.pth" in flat_source
    assert "_set_tolerant_term(self, \"anchor_pos_z\", 0.5)" in flat_source
    assert "_set_tolerant_term(self, \"ee_body_pos_z\", 0.4)" in flat_source
    assert "_set_tolerant_term(self, \"anchor_ori\", 0.6)" in flat_source
    assert '"name": "hip_dof"' in flat_source
    assert '"func": "bad_hip_dof"' in flat_source
    assert "math.pi / 6.0" in flat_source
    assert "G1_STAGE_III_ISAAC_EQUIVALENCE_NOTES" in flat_source
    assert "reset_events" in flat_source
    assert "self.events.reset_base = EventTerm" in flat_source
    assert "func=mdp.reset_root_state_uniform" in flat_source
    assert "self.events.reset_robot_joints = EventTerm" in flat_source
    assert "func=mdp.reset_joints_by_offset" in flat_source
    assert '"sensor_cfg": SceneEntityCfg(' in flat_source
    assert "left_ankle_roll_link$)(?!right_ankle_roll_link$)" in flat_source

    assert "actor_hidden_dims=[512, 256, 128]" in agent_source
    assert "critic_hidden_dims=[512, 256, 128]" in agent_source
    assert "learning_rate=1.0e-3" in agent_source
    assert "entropy_coef=0.005" in agent_source
    assert "max_iterations = 30001" in agent_source
    assert "experiment_name = \"g1_tracking\"" in agent_source


def test_shared_mdp_supports_g1_root_standing_lke_and_tolerant_termination():
    commands_source = COMMANDS_PATH.read_text()
    rewards_source = REWARDS_PATH.read_text()
    terminations_source = TERMINATIONS_PATH.read_text()

    assert "sampling_mode: Literal[\"adaptive\", \"uniform\", \"start\", \"lke\"]" in commands_source
    assert "class MotionStandingCommand(MotionCommand)" in commands_source
    assert "root_body_name: str | None = None" in commands_source
    assert "robot_root_states_xyzw" in commands_source
    assert "Standing init dof_pos must have shape" in commands_source
    assert "standing_joint_pos.shape[1] == joint_pos.shape[1]" not in commands_source
    assert "select_diverse_grsi_pool_indexes" in commands_source
    assert "torch.randperm" not in commands_source
    assert "kinetic_energy_prob" in commands_source
    assert "lke_anchor_weights" in commands_source
    assert "def penalty_relative_shoulder_high" in rewards_source
    assert "def penalty_relative_root_orientation" in rewards_source
    assert "def penalty_xy_rate_before_stand" in rewards_source
    assert "def penalty_action_rate_before_stand" in rewards_source
    assert "def penalty_electrical_power_cost" in rewards_source
    assert "def reward_center_of_mass" in rewards_source
    assert "class TolerantTermination" in terminations_source
    assert "def bad_hip_dof" in terminations_source
    assert "net_forces_w_history" in rewards_source
    assert "sensor_cfg: SceneEntityCfg" in rewards_source


def test_g1_action_scale_is_expanded_for_concrete_joint_names():
    robot_source = G1_ROBOT_PATH.read_text()
    g1_mdp = _load_g1_mdp()
    module = ast.parse(robot_source)
    robot_joint_names = None
    for node in module.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "G1_JOINT_NAMES" for target in node.targets):
            robot_joint_names = ast.literal_eval(node.value)
            break

    assert robot_joint_names == g1_mdp.G1_MOTION_JOINT_NAMES
    assert "G1_JOINT_NAMES = [" in robot_source
    assert "re.fullmatch(pattern, joint_name)" in robot_source
    assert "G1_ACTION_SCALE[joint_name]" in robot_source


def test_replay_npz_accepts_g1_and_uses_g1_joint_order():
    replay_source = REPLAY_PATH.read_text()

    assert 'choices=["pm01", "t800", "g1"]' in replay_source
    assert "from whole_body_tracking.robots.g1 import G1_CFG" in replay_source
    assert "G1_MOTION_JOINT_NAMES" in replay_source
    assert "def assemble_replay_root_states" in replay_source
    assert "motion_root_pos_w + env_origins" in replay_source
    assert "scene.env_origins[:, None, :]" not in replay_source
    assert re.search(r'"g1":\s*G1_CFG', replay_source)
    assert re.search(r'"g1":\s*G1_MOTION_JOINT_NAMES', replay_source)
