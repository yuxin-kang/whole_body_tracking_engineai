"""Convert headerless root XYZ + quaternion XYZW + 25 SDK joints to tracking NPZ."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from convert_t800_getup_to_npz import (
    DEFAULT_URDF, _normalize_quaternions, _linear_interpolate,
    _slerp_interpolate, _build_body_kinematics, _angular_velocity,
    CANONICAL_BODY_NAMES, load_urdf_tree,
)


def playback_times(duration, fps, start_hold=0., end_hold=0., ramp=0.):
    """Map output time to source time, easing playback speed at both ends.

    Each ramp integrates smoothstep speed from zero to one. Spatial poses
    remain on the source path; velocities must be recomputed after resampling.
    """
    if duration <= 0 or fps <= 0 or min(start_hold, end_hold, ramp) < 0:
        raise ValueError('Invalid duration, fps, hold or ramp')
    if ramp > duration:
        raise ValueError('Speed ramp must not exceed source duration')
    active_duration = duration + ramp
    output = np.arange(int(np.ceil((start_hold + active_duration + end_hold) * fps)) + 1) / fps
    active = np.clip(output - start_hold, 0., active_duration)
    source = active.copy()
    if ramp:
        source = active - ramp / 2
        u = np.clip(active / ramp, 0., 1.)
        v = np.clip((active_duration - active) / ramp, 0., 1.)
        source = np.where(active < ramp, ramp * (u**3 - 0.5*u**4), source)
        source = np.where(active > active_duration - ramp,
                          duration - ramp * (v**3 - 0.5*v**4), source)
    return np.clip(source, 0., duration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-fps', type=float, required=True)
    parser.add_argument('--output-fps', type=float, default=50.)
    parser.add_argument('--trim-start-frames', type=int, default=0)
    parser.add_argument('--start-hold', type=float, default=0.)
    parser.add_argument('--end-hold', type=float, default=0.)
    parser.add_argument('--speed-ramp', type=float, default=0.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = np.loadtxt(args.input, delimiter=',', ndmin=2)
    if raw.shape[1] != 32 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError(f'Expected finite [frames>=2,32], got {raw.shape}')
    if not 0 < args.input_fps < 10000 or not 0 < args.output_fps < 10000:
        raise ValueError('Invalid frame rate')
    original_frames = len(raw)
    if not 0 <= args.trim_start_frames <= len(raw) - 2:
        raise ValueError('Trimming must retain at least two source frames')
    raw = raw[args.trim_start_frames:]
    source_names = [f'{part}_{side}' for side in ('L', 'R')
                    for part in ('HIP_PITCH','HIP_ROLL','HIP_YAW','KNEE_PITCH','ANKLE_PITCH','ANKLE_ROLL')]
    source_names += ['TORSO_YAW']
    source_names += [f'{part}_{side}' for side in ('L','R')
                     for part in ('SHOULDER_PITCH','SHOULDER_ROLL','SHOULDER_YAW','ELBOW_PITCH','ELBOW_YAW')]
    source_names += ['HEAD_PITCH','HEAD_YAW']
    joint_names = [j.attrib['name'] for j in ET.parse(DEFAULT_URDF).getroot().findall('joint') if j.attrib['type'] != 'fixed']
    order = [source_names.index(n.split('_',1)[1]) for n in joint_names]
    times = np.arange(len(raw)) / args.input_fps
    target = playback_times(times[-1], args.output_fps, args.start_hold, args.end_hold, args.speed_ramp)
    root = raw[:,:3].copy()
    # Only translate horizontal origin: retain heights, heading and motion timing.
    offset = root[0,:2].copy()
    root[:,:2] -= offset
    root = _linear_interpolate(root, times, target)
    quat = _slerp_interpolate(_normalize_quaternions(raw[:,[6,3,4,5]]), times, target)
    q = _linear_interpolate(raw[:,7:][:,order], times, target)
    base, joints = load_urdf_tree(DEFAULT_URDF)
    pos, ori = _build_body_kinematics(base, joints, root, quat, q)
    payload = dict(fps=np.array([args.output_fps]), joint_pos=q,
                   joint_vel=np.gradient(q, 1/args.output_fps, axis=0),
                   body_pos_w=pos, body_quat_w=ori,
                   body_lin_vel_w=np.gradient(pos,1/args.output_fps,axis=0),
                   body_ang_vel_w=_angular_velocity(ori,1/args.output_fps))
    payload = {k:v.astype(np.float32) for k,v in payload.items()}
    assert all(np.isfinite(v).all() for v in payload.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    report = dict(source=str(args.input), sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  source_frames=original_frames, retained_source_frames=len(raw),
                  trim_start_frames=args.trim_start_frames, trim_start_s=args.trim_start_frames / args.input_fps,
                  start_hold_s=args.start_hold, end_hold_s=args.end_hold, speed_ramp_s=args.speed_ramp,
                  output_frames=len(q), source_fps=args.input_fps,
                  output_fps=args.output_fps, duration_s=float(times[-1]),
                  output_duration_s=(len(q)-1) / args.output_fps,
                  xy_origin_removed=offset.tolist(), joint_names=joint_names,
                  body_names=CANONICAL_BODY_NAMES, height_correction=False,
                  root_height_range=[float(root[:,2].min()),float(root[:,2].max())])
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
