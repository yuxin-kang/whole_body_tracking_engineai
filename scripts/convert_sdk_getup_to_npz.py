#!/usr/bin/env python3
"""Export SDK get-up references to the existing T800 tracking schema (CPU only)."""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from convert_t800_getup_to_npz import (
    DEFAULT_URDF, CANONICAL_BODY_NAMES, _normalize_quaternions,
    _build_body_kinematics, _linear_interpolate, _slerp_interpolate,
    _output_times, _angular_velocity, load_urdf_tree,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, default=Path(
        '/srv/shared/home/xjh/ws_kyx/engineai_robotics_native_sdk'))
    parser.add_argument('--output', type=Path, default=Path('data/npz/t800_sdk_get_up'))
    args = parser.parse_args()
    root, tree = load_urdf_tree(DEFAULT_URDF)
    joint_names = [j.attrib['name'] for j in ET.parse(DEFAULT_URDF).getroot().findall('joint')
                   if j.attrib['type'] != 'fixed']
    config_root = args.sdk / 'assets/config/t800'
    args.output.mkdir(parents=True, exist_ok=True)
    for kind in ('supine', 'prone'):
        config_path = config_root / f'rl_{kind}_to_stance/default.yaml'
        config = yaml.safe_load(config_path.read_text())
        state = config['motion_state']
        source = config_root / state['data_path']
        if kind == 'supine':
            raw = np.load(source, allow_pickle=False).astype(np.float64)
            assert raw.ndim == 2 and raw.shape[1] == 34, raw.shape
            pos, quat = raw[:, :3], raw[:, 3:7]
            # This 34-column asset stores wxyz: its standing endpoints have
            # upright base Z under wxyz, but almost horizontal Z under xyzw.
            sdk_names = {n.split('_', 1)[1]: i for i, n in enumerate(config['active_joint_names'])}
            order = [sdk_names[n.split('_', 1)[1]] for n in joint_names]
            joints = raw[:, 7:32][:, order]
        else:
            raw = np.genfromtxt(source, delimiter=',', names=True)
            pos = np.column_stack([raw[f'LINK_BASE_p{a}'] for a in 'xyz'])
            quat = np.column_stack([raw[f'LINK_BASE_q{a}'] for a in 'wxyz'])
            csv_names = {n.split('_', 1)[1]: n for n in raw.dtype.names if n.startswith('J')}
            joints = np.column_stack([raw[csv_names[n.split('_', 1)[1]]] for n in joint_names])
        assert all(np.isfinite(a).all() for a in (pos, quat, joints))
        quat = _normalize_quaternions(quat)
        frame_count = len(pos)
        start, end = state['traj_frame']
        start = min(frame_count - 1, start) if start >= 0 else max(0, frame_count + start)
        end = min(frame_count - 1, end) if end >= 0 else max(0, frame_count + end)
        for variant, first, last in [('full', 0, frame_count - 1), ('sdk_clip', start, end)]:
            output = args.output / f'T800_{kind}_to_stance_{variant}_50hz.npz'
            if output.exists():
                raise FileExistsError(f'Refusing to replace existing export: {output}')
            assert last > first
            sl = slice(first, last + 1)
            # Preserve the configured SDK sample interval, including rounding.
            source_dt = float(state['csv_dt'])
            source_times = np.arange(last - first + 1) * source_dt
            times = _output_times(last - first + 1, 1 / source_dt, 50)
            q = _linear_interpolate(joints[sl], source_times, times)
            p = _linear_interpolate(pos[sl], source_times, times)
            r = _slerp_interpolate(quat[sl], source_times, times)
            bp, bq = _build_body_kinematics(root, tree, p, r, q)
            arrays = dict(fps=np.array([50], dtype=np.float32), joint_pos=q,
                          joint_vel=np.gradient(q, .02, axis=0), body_pos_w=bp,
                          body_quat_w=bq, body_lin_vel_w=np.gradient(bp, .02, axis=0),
                          body_ang_vel_w=_angular_velocity(bq, .02))
            arrays = {k: v.astype(np.float32) for k, v in arrays.items()}
            assert all(np.isfinite(v).all() for v in arrays.values())
            assert np.allclose(np.linalg.norm(bq, axis=-1), 1, atol=1e-6)
            assert np.allclose(q[[0, -1]], joints[[first, last]])
            np.savez_compressed(output, **arrays)
            metadata = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            source_config=str(config_path), source_dt=source_dt,
                            source_frames_inclusive=[first, last], source_total_frames=frame_count,
                            output_frames=len(q), output_fps=50, duration_s=(len(q)-1)/50,
                            joint_names=joint_names, body_names=CANONICAL_BODY_NAMES,
                            quaternion_order='wxyz', body_state_method='current T800 URDF forward kinematics',
                            urdf=str(DEFAULT_URDF), urdf_sha256=hashlib.sha256(DEFAULT_URDF.read_bytes()).hexdigest(),
                            note='Reference trajectory only; not an MNN policy rollout. No extra terminal hold.',
                            output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
            print(f'{output}: {len(q)} frames, {(len(q)-1)/50:.2f}s; source [{first}, {last}]')


if __name__ == '__main__':
    main()
