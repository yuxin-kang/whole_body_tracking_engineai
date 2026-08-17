#!/usr/bin/env python3
"""Validate T800 motion body positions against URDF forward kinematics."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Joint:
    parent: str
    child: str
    origin_position: np.ndarray
    origin_rotation: np.ndarray
    axis: np.ndarray | None
    motion_index: int | None


def _vector(element: ET.Element | None, attribute: str, default: str) -> np.ndarray:
    value = default if element is None else element.attrib.get(attribute, default)
    return np.fromstring(value, sep=" ", dtype=np.float64)


def load_urdf_tree(path: Path) -> tuple[str, list[Joint]]:
    robot = ET.parse(path).getroot()
    links = {element.attrib["name"] for element in robot.findall("link")}
    child_links: set[str] = set()
    joints: list[Joint] = []
    motion_index = 0

    for element in robot.findall("joint"):
        parent = element.find("parent").attrib["link"]
        child = element.find("child").attrib["link"]
        child_links.add(child)
        origin = element.find("origin")
        origin_position = _vector(origin, "xyz", "0 0 0")
        origin_rpy = _vector(origin, "rpy", "0 0 0")
        origin_rotation = Rotation.from_euler("xyz", origin_rpy).as_matrix()

        if element.attrib["type"] == "fixed":
            axis = None
            index = None
        else:
            axis = _vector(element.find("axis"), "xyz", "1 0 0")
            axis /= np.linalg.norm(axis)
            index = motion_index
            motion_index += 1

        joints.append(
            Joint(parent, child, origin_position, origin_rotation, axis, index)
        )

    roots = links - child_links
    if len(roots) != 1:
        raise ValueError(f"Expected one URDF root link, found {sorted(roots)}")
    if motion_index != 25:
        raise ValueError(f"Expected 25 actuated joints, found {motion_index}")
    return roots.pop(), joints


def forward_kinematics(
    root_link: str,
    joints: list[Joint],
    root_position: np.ndarray,
    root_quaternion_wxyz: np.ndarray,
    joint_position: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root_rotation = Rotation.from_quat(root_quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()
    transforms = {root_link: (root_position.astype(np.float64), root_rotation)}

    unresolved = list(joints)
    while unresolved:
        next_unresolved: list[Joint] = []
        for joint in unresolved:
            if joint.parent not in transforms:
                next_unresolved.append(joint)
                continue

            parent_position, parent_rotation = transforms[joint.parent]
            child_position = parent_position + parent_rotation @ joint.origin_position
            child_rotation = parent_rotation @ joint.origin_rotation
            if joint.motion_index is not None:
                joint_rotation = Rotation.from_rotvec(
                    joint.axis * joint_position[joint.motion_index]
                ).as_matrix()
                child_rotation = child_rotation @ joint_rotation
            transforms[joint.child] = (child_position, child_rotation)

        if len(next_unresolved) == len(unresolved):
            missing = sorted(joint.child for joint in next_unresolved)
            raise ValueError(f"URDF joint graph is disconnected: {missing}")
        unresolved = next_unresolved

    return transforms


def validate_motion(path: Path, urdf_path: Path, tolerance: float) -> tuple[float, float]:
    root_link, joints = load_urdf_tree(urdf_path)
    with np.load(path, allow_pickle=False) as motion:
        required = {"joint_pos", "body_pos_w", "body_quat_w", "body_names"}
        missing = required - set(motion.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")

        joint_position = np.asarray(motion["joint_pos"])
        body_position = np.asarray(motion["body_pos_w"])
        body_quaternion = np.asarray(motion["body_quat_w"])
        body_names = tuple(str(name) for name in motion["body_names"].tolist())

    checked_indices = [index for index, name in enumerate(body_names) if name == root_link]
    urdf_children = {joint.child for joint in joints}
    checked_indices.extend(
        index for index, name in enumerate(body_names) if name in urdf_children
    )

    errors: list[float] = []
    for frame in range(joint_position.shape[0]):
        root_index = body_names.index(root_link)
        transforms = forward_kinematics(
            root_link,
            joints,
            body_position[frame, root_index],
            body_quaternion[frame, root_index],
            joint_position[frame],
        )
        for body_index in checked_indices:
            name = body_names[body_index]
            expected_position = transforms[name][0]
            errors.append(float(np.linalg.norm(body_position[frame, body_index] - expected_position)))

    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    if max_error > tolerance:
        raise ValueError(
            f"{path}: FK position error exceeds {tolerance:g} m "
            f"(mean={mean_error:.8g}, max={max_error:.8g})"
        )
    return mean_error, max_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motions", nargs="+", type=Path)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path(
            "source/whole_body_tracking/whole_body_tracking/assets/t800/urdf/serial_t800.urdf"
        ),
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for motion in args.motions:
        mean_error, max_error = validate_motion(motion, args.urdf, args.tolerance)
        print(f"[OK] {motion}: mean={mean_error:.8g} m, max={max_error:.8g} m")


if __name__ == "__main__":
    main()
