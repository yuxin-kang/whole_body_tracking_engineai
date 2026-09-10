"""Lift reference frames only as needed to keep T800 foot collision boxes above ground."""
import argparse
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from convert_t800_getup_to_npz import CANONICAL_BODY_NAMES, DEFAULT_URDF


def foot_bottoms(arrays):
    robot = ET.parse(DEFAULT_URDF).getroot()
    bottoms = []
    for name in ('LINK_ANKLE_ROLL_L', 'LINK_ANKLE_ROLL_R'):
        link = robot.find(f"link[@name='{name}']")
        idx = CANONICAL_BODY_NAMES.index(name)
        rot = Rotation.from_quat(arrays['body_quat_w'][:, idx][:, [1, 2, 3, 0]]).as_matrix()
        for collision in link.findall('collision'):
            box = collision.find('geometry/box')
            if box is None:
                raise ValueError('Expected box foot collision')
            half = np.fromstring(box.attrib['size'], sep=' ') / 2
            origin = collision.find('origin')
            xyz = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')
            corners = np.array(list(itertools.product((-1, 1), repeat=3))) * half
            corners = Rotation.from_euler('xyz', rpy).apply(corners) + xyz
            world = np.einsum('nij,kj->nki', rot, corners) + arrays['body_pos_w'][:, idx, None]
            bottoms.append(world[:, :, 2].min(axis=1))
    return np.stack(bottoms).min(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.source, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    before = foot_bottoms(arrays)
    lift = np.maximum(0., .002 - before)
    arrays['body_pos_w'][:, :, 2] += lift[:, None]
    dt = 1 / float(arrays['fps'].reshape(-1)[0])
    arrays['body_lin_vel_w'] = np.gradient(arrays['body_pos_w'], dt, axis=0).astype(np.float32)
    after = foot_bottoms(arrays)
    assert after.min() >= .002 - 1e-6
    assert all(np.isfinite(a).all() for a in arrays.values())
    np.savez_compressed(args.output, **arrays)
    report = dict(source=str(args.source), foot_min_z_before_m=float(before.min()),
                  foot_min_z_after_m=float(after.min()), max_lift_m=float(lift.max()),
                  method='Per-frame vertical translation to foot collision floor clearance of 2 mm',
                  limitation='Geometric replay correction only. Hand contacts, visual meshes and dynamic feasibility not validated.')
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
