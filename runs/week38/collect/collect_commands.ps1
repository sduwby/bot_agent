# ===== 多任务采集清单（**逐条执行**；录制是交互式的：每集按 →/Enter 结束，坏集按 ← 重录）=====
# 采完每条后：跑一次 --verify，通过再进入下一个 repo。
# 物体位置要**随机覆盖**左/中/右、近/远（固定轨迹会让策略只会背路径）。
# 解释器（必须装了 lerobot）：D:\miniconda\envs\lerobot\python.exe
# 旁挂负载：每个 repo 一份（目录 runs\week38\collect\sidecars），由 sidecar_for() 自动生成。
# 数据集本地根：C:\Users\Administrator\.cache\huggingface\lerobot\<repo 名>（--verify 的 --dataset-root 已按此填好）。
# 断开时卸力矩：False（不卸：避开夹爪过载时的 Overload error；退出后舵机持续通电保持姿态）
# 上传：push_to_hub=False（只落本地；铺满后再用 --push-to-hub 重新生成）

# ⚠ 新建数据集要求该 repo 的本地目录**不存在**（即上面的数据集本地根）。
#   中断后重跑若报 FileExistsError：先看该目录——**只有 meta/info.json 就是空壳，删掉重跑**；
#   已有 episode 则改用续录（**顶层**参数，不是 dataset 子参数）：在录制命令末尾加 `--resume=true`，
#   **别删**该目录（删了就丢已录数据）；续录时旁挂文件也**不要删**，会追加到同一份。

# ---- [1/6] 物体「螺丝刀」 × pick_a → 夹起桌面上的螺丝刀 （20 集）----
& "D:\miniconda\envs\lerobot\python.exe" src\run_vla_collect_record.py `
  --sidecar runs\week38\collect\sidecars\demo_vla_01_pick_a.jsonl `
  --robot.type=so101_follower --robot.port=COM6 --robot.id=my_awesome_follower_arm `
  --robot.cameras='{"fixed": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}, "handeye": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' `
  --robot.disable_torque_on_disconnect=false `
  --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_awesome_leader_arm `
  --dataset.repo_id=wangbangyan/demo_vla_01_pick_a `
  --dataset.single_task="夹起桌面上的螺丝刀" `
  --dataset.num_episodes=20 --dataset.episode_time_s=99999 --dataset.reset_time_s=99999 `
  --dataset.push_to_hub=false --display_data=false
# 采完校验（旁挂 vs 数据集逐帧对齐；通过再进下一个 repo）：
& "D:\miniconda\envs\lerobot\python.exe" src\run_vla_collect_record.py --verify `
  --dataset-root C:\Users\Administrator\.cache\huggingface\lerobot\wangbangyan\demo_vla_01_pick_a `
  --sidecar runs\week38\collect\sidecars\demo_vla_01_pick_a.jsonl

# ---- [2/6] 物体「螺丝刀」 × place_left → 把螺丝刀放到左边 （20 集）----
& "D:\miniconda\envs\lerobot\python.exe" src\run_vla_collect_record.py `
  --sidecar runs\week38\collect\sidecars\demo_vla_01_place_left.jsonl `
  --robot.type=so101_follower --robot.port=COM6 --robot.id=my_awesome_follower_arm `
  --robot.cameras='{"fixed": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}, "handeye": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' `
  --robot.disable_torque_on_disconnect=false `
  --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_awesome_leader_arm `
  --dataset.repo_id=wangbangyan/demo_vla_01_place_left `
  --dataset.single_task="把螺丝刀放到左边" `
  --dataset.num_episodes=20 --dataset.episode_time_s=99999 --dataset.reset_time_s=99999 `
  --dataset.push_to_hub=false --display_data=false
# 采完校验（旁挂 vs 数据集逐帧对齐；通过再进下一个 repo）：
& "D:\miniconda\envs\lerobot\python.exe" src\run_vla_collect_record.py --verify `
  --dataset-root C:\Users\Administrator\.cache\huggingface\lerobot\wangbangyan\demo_vla_01_place_left `
  --sidecar runs\week38\collect\sidecars\demo_vla_01_place_left.jsonl

# ---- [3/6] 物体「黑色长条」 × pick_a → 夹起桌面上的黑色长条 （20 集）----
& "D:\miniconda\envs\lerobot\python.exe" src\run_vla_collect_record.py `
  --sidecar runs\week38\collect\sidecars\demo_vla_02_pick_a.jsonl `
  --robot.type=so101_follower --robot.port=COM6 --robot.id=my_awesome_follower_arm `
  --robot.cameras='{"fixed": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}, "handeye": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' `
  --robot.disable_torque_on_disconnect=false `
  --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_awesome_leader_arm `
  --dataset.repo_id=wangbangyan/demo_vla_02_pick_a `
  --dataset.single_task="夹起桌面上的黑色长条" `
  --dataset.num_episodes=20 --dataset.episode_time_s=99999 --dataset.reset_time_s=99999 `
  --dataset.push_to_hub=false --display_data=false
# 采完校验（旁挂 vs 数据集逐帧对齐；通过再进下一个 repo）：
& "D:\miniconda\envs\lerobot\python.exe" src\run_vla_collect_record.py --verify `
  --dataset-root C:\Users\Administrator\.cache\huggingface\lerobot\wangbangyan\demo_vla_02_pick_a `
  --sidecar runs\week38\collect\sidecars\demo_vla_02_pick_a.jsonl

# ---- [4/6] 物体「黑色长条」 × place_left → 把黑色长条放到左边 （20 集）----
& "D:\miniconda\envs\lerobot\python.exe" src\run_vla_collect_record.py `
  --sidecar runs\week38\collect\sidecars\demo_vla_02_place_left.jsonl `
  --robot.type=so101_follower --robot.port=COM6 --robot.id=my_awesome_follower_arm `
  --robot.cameras='{"fixed": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}, "handeye": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' `
  --robot.disable_torque_on_disconnect=false `
  --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_awesome_leader_arm `
  --dataset.repo_id=wangbangyan/demo_vla_02_place_left `
  --dataset.single_task="把黑色长条放到左边" `
  --dataset.num_episodes=20 --dataset.episode_time_s=99999 --dataset.reset_time_s=99999 `
  --dataset.push_to_hub=false --display_data=false
# 采完校验（旁挂 vs 数据集逐帧对齐；通过再进下一个 repo）：
& "D:\miniconda\envs\lerobot\python.exe" src\run_vla_collect_record.py --verify `
  --dataset-root C:\Users\Administrator\.cache\huggingface\lerobot\wangbangyan\demo_vla_02_place_left `
  --sidecar runs\week38\collect\sidecars\demo_vla_02_place_left.jsonl

# ---- [5/6] 物体「指甲刀」 × pick_a → 夹起桌面上的指甲刀 （20 集）----
& "D:\miniconda\envs\lerobot\python.exe" src\run_vla_collect_record.py `
  --sidecar runs\week38\collect\sidecars\demo_vla_03_pick_a.jsonl `
  --robot.type=so101_follower --robot.port=COM6 --robot.id=my_awesome_follower_arm `
  --robot.cameras='{"fixed": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}, "handeye": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' `
  --robot.disable_torque_on_disconnect=false `
  --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_awesome_leader_arm `
  --dataset.repo_id=wangbangyan/demo_vla_03_pick_a `
  --dataset.single_task="夹起桌面上的指甲刀" `
  --dataset.num_episodes=20 --dataset.episode_time_s=99999 --dataset.reset_time_s=99999 `
  --dataset.push_to_hub=false --display_data=false
# 采完校验（旁挂 vs 数据集逐帧对齐；通过再进下一个 repo）：
& "D:\miniconda\envs\lerobot\python.exe" src\run_vla_collect_record.py --verify `
  --dataset-root C:\Users\Administrator\.cache\huggingface\lerobot\wangbangyan\demo_vla_03_pick_a `
  --sidecar runs\week38\collect\sidecars\demo_vla_03_pick_a.jsonl

# ---- [6/6] 物体「指甲刀」 × place_left → 把指甲刀放到左边 （20 集）----
& "D:\miniconda\envs\lerobot\python.exe" src\run_vla_collect_record.py `
  --sidecar runs\week38\collect\sidecars\demo_vla_03_place_left.jsonl `
  --robot.type=so101_follower --robot.port=COM6 --robot.id=my_awesome_follower_arm `
  --robot.cameras='{"fixed": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}, "handeye": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' `
  --robot.disable_torque_on_disconnect=false `
  --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_awesome_leader_arm `
  --dataset.repo_id=wangbangyan/demo_vla_03_place_left `
  --dataset.single_task="把指甲刀放到左边" `
  --dataset.num_episodes=20 --dataset.episode_time_s=99999 --dataset.reset_time_s=99999 `
  --dataset.push_to_hub=false --display_data=false
# 采完校验（旁挂 vs 数据集逐帧对齐；通过再进下一个 repo）：
& "D:\miniconda\envs\lerobot\python.exe" src\run_vla_collect_record.py --verify `
  --dataset-root C:\Users\Administrator\.cache\huggingface\lerobot\wangbangyan\demo_vla_03_place_left `
  --sidecar runs\week38\collect\sidecars\demo_vla_03_place_left.jsonl
