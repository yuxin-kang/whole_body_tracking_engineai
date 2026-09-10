"""Check calibrated walls against the full reference and active Isaac URDF.

This is a conservative geometry check, not a learned-policy rollout. Run from
the repository root; optional --output stores the JSON evidence for a run.
"""

import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "source/whole_body_tracking/whole_body_tracking/assets/t800/urdf/serial_t800.urdf"
CONFIG = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800/mixed_wall_env_cfg.py"
SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)))
WALL_HALF_SIZE = np.array([.09, .7, .8])


def explicit_spec(name):
    assignment = next(n for n in ast.parse(CONFIG.read_text()).body
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return {ast.literal_eval(k): ast.literal_eval(v)
            for k, v in zip(assignment.value.keys, assignment.value.values) if k is not None}


def origin(element):
    if element is None:
        return np.zeros(3), np.eye(3)
    return (np.fromstring(element.get("xyz", "0 0 0"), sep=" "),
            Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix())


def analyze(motion_path, strike, spec, old_center, old_yaw):
    tree = ET.parse(URDF).getroot()
    with np.load(motion_path, allow_pickle=False) as motion:
        poses = {name: (motion["body_pos_w"][:, i],
                       Rotation.from_quat(motion["body_quat_w"][:, i, [1, 2, 3, 0]]).as_matrix())
                 for i, name in enumerate(motion["body_names"].tolist())}
        frames = len(motion["joint_pos"])
    # Resolve fixed wrist/foot links omitted from the 34-body motion schema.
    for _ in range(len(tree.findall("joint"))):
        for joint in tree.findall("joint"):
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if joint.get("type") == "fixed" and child not in poses and parent in poses:
                offset, local_rot = origin(joint.find("origin"))
                pos, rot = poses[parent]
                poses[child] = (pos + np.einsum("fij,j->fi", rot, offset), rot @ local_rot)
    hits = {}
    strike_corners = None
    for link in tree.findall("link"):
        name = link.get("name")
        for collision in link.findall("collision"):
            if name not in poses:
                raise ValueError(f"Missing collision-link pose: {name}")
            geom = collision.find("geometry")
            if geom.find("box") is not None:
                half = np.fromstring(geom.find("box").get("size"), sep=" ") / 2
            elif geom.find("sphere") is not None:
                half = np.repeat(float(geom.find("sphere").get("radius")), 3)
            elif geom.find("cylinder") is not None:
                cylinder = geom.find("cylinder")
                half = np.array([float(cylinder.get("radius"))] * 2 + [float(cylinder.get("length")) / 2])
            else:
                raise ValueError(f"Unsupported collision geometry: {name}")
            offset, local_rot = origin(collision.find("origin"))
            pos, rot = poses[name]
            corners = np.einsum("fij,kj->fki", rot, (SIGNS * half) @ local_rot.T + offset) + pos[:, None, :]
            # New walls both have yaw=0. Curved shapes use conservative boxes.
            local = corners - spec["target_position"]
            overlaps = np.all((local.max(1) >= -WALL_HALF_SIZE) & (local.min(1) <= WALL_HALF_SIZE), axis=1)
            if overlaps.any():
                hits.setdefault(name, set()).update(np.flatnonzero(overlaps).tolist())
            if name == strike:
                if strike_corners is not None:
                    raise ValueError("Expected the active Isaac URDF's single foot box")
                strike_corners = corners
    assert set(hits) == {strike}, f"Unexpected/missing wall contacts: {hits}"
    leading_x = strike_corners[:, :, 0].max(1)
    penetration = leading_x.max() - (spec["target_position"][0] - WALL_HALF_SIZE[0])
    assert .02 <= penetration <= .03, f"Expected 2.5 cm shallow overlap; got {penetration}"
    assert all(spec["contact_frames"][0] <= f <= spec["contact_frames"][1] for f in hits[strike])
    old_normal = np.array([np.cos(old_yaw), np.sin(old_yaw), 0.])
    old_plane_crossing = ((strike_corners - old_center) @ old_normal).max() + WALL_HALF_SIZE[0]
    return {
        "motion": str(motion_path.relative_to(ROOT)), "frames": frames,
        "motion_sha256": hashlib.sha256(motion_path.read_bytes()).hexdigest(),
        "new_wall_center": spec["target_position"], "new_wall_yaw": 0.,
        "near_face_x": spec["target_position"][0] - WALL_HALF_SIZE[0],
        "max_reference_foot_x": float(leading_x.max()), "max_extension_frame": int(leading_x.argmax()),
        "max_nominal_penetration_m": float(penetration),
        "old_max_crossing_of_infinite_front_plane_m": float(old_plane_crossing),
        "conservative_collision_frames": {name: sorted(values) for name, values in hits.items()},
        "contact_reward_frames": spec["contact_frames"], "non_strike_obstructions": [],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = {
        "scope": "full reference geometry, conservative primitive bounds; not a learned-policy evaluation",
        "urdf": str(URDF.relative_to(ROOT)), "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "left_front_kick": analyze(
            ROOT / "data/npz/traj_eng_50hz_episode_complete/left_front_kick_002_terminal_hold_1s.npz",
            "LINK_ANKLE_ROLL_L", explicit_spec("FRONT_KICK_MIXED_WALL_SPEC"), [1.821, .034, 1.135], 0.),
        "roundhouse_kick": analyze(
            ROOT / "data/npz/traj_eng_50hz_improved/roundhouse_kick_001_retimed_terminal_hold_1s.npz",
            "LINK_ANKLE_ROLL_R", explicit_spec("ROUNDHOUSE_MIXED_WALL_SPEC"), [1.330, -.200, 1.570], -1.33),
    }
    result = json.dumps(evidence, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result)
