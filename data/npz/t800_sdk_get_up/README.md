# SDK 起身轨迹（50 Hz tracking 格式）

优先用于单次起身训练：

| 文件 | 原始零基帧范围（含结束帧） | 输出帧数 | 时长 |
| --- | --- | --- | --- |
| T800_supine_to_stance_sdk_clip_50hz.npz | 90–180 | 151 | 3.00 s |
| T800_prone_to_stance_sdk_clip_50hz.npz | 10–95 | 143 | 2.84 s |

`full` 文件保留整个源轨迹：仰卧 328 帧 / 6.54 s，俯卧 250 帧 / 4.98 s。
仰卧完整版包含开始站立和倒地过程，不等同于从倒地开始的起身片段。

每份 NPZ 仅有七个训练键：`fps`、`joint_pos`、`joint_vel`、`body_pos_w`、
`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`。
关节采用当前 T800 URDF 的 25 关节顺序，身体采用现有 get-up 的 34-body canonical 顺序，
四元数为 wxyz。源采样间隔为 SDK 配置的 0.033333 s，输出为 50 Hz。
关节角和根部位置线性插值，根部旋转 SLERP；身体位姿通过当前 URDF 正向运动学重建，速度由差分计算。
未额外添加末尾站立时间。原始文件保持不变，来源、哈希、顺序和裁剪范围在同名 JSON 中。

这些是参考动作数据，并非 MNN 策略实际运行产生的轨迹，也不是可直接恢复训练的策略权重。
完成了数据结构、有限值、四元数归一化和运动学一致性检查；尚未在 IsaacLab 中回放或训练。

在训练命令中用相应路径替换 `--motion_file` 即可指定轨迹。仰卧初始姿态与原先俯卧不同，
训练前仍应确认所选任务的初始状态和阶段奖励适合该动作。

可复现转换：

```bash
/srv/shared/home/xjh/app/miniconda3/envs/whole_body_tracking/bin/python scripts/convert_sdk_getup_to_npz.py --output /path/to/new/output
```
