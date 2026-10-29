#!/usr/bin/env python
r"""多任务采集编排：把「物体 × 指令」展开成**可执行的采集任务清单**。

## 为什么需要它

`lerobot-record` 的 `--dataset.single_task` **一次只支持一条指令文本**，所以"多任务数据集"
实际上是**多个单任务 repo 的集合**。手工展开 3 物体 × 3 指令很容易在"repo 名 / 指令文本 / 物体"
三者之间错位，而这类错位会一路传到下游：P2 的 `(object, instruction)` 组合、P1 的意图标注。
本脚本把展开规则固化，并输出三样东西：

- `collect_tasks.json`：任务清单（含每个 repo 对应的物体与指令全文）
- `collect_combos.json`：`[[object, instruction], ...]`，直接喂给
  `run_vla_generalization.py --combos-json`（combo/paraphrase/novel_object 三协议都靠它）
- `collect_commands.ps1`：逐条可跑的录制命令 + 采完的 `--verify` 命令

**旁挂负载每个 repo 一份**（`sidecars/<repo 名>.jsonl`）：`LoadSidecarWriter` 是追加写、
`--verify` 又把整份旁挂对**单个**数据集根目录逐帧比对，共用一个文件会让第 2 个 repo 起校验失败。

`--verify` 的 `--dataset-root` **已按 `LEROBOT_CACHE_ROOT` 逐 repo 填好实际路径**（不留占位符），
即 `<缓存根>\<repo_id>`；录制命令也固定用装了 lerobot 的解释器（`PYTHON`）。两者都是实跑踩过的坑。

## 三条指令为什么要这样配（决定下游实验能不能跑）

| key | 文本模板 | 作用 |
|---|---|---|
| `pick_a` | 夹起桌面上的{obj} | 意图 pick，措辞 A |
| `pick_b` | 把{obj}拿起来 | 意图 pick，措辞 B —— **pick_a 的同义改写**，供 P2 的 `paraphrase` 协议（**可选**，见下） |
| `place_left` | 把{obj}放到左边 | 意图 place —— **提供第二个意图**，否则意图只有一类、A3 的 κ 是退化值 |

`pick_b` 的**动作与 `pick_a` 完全相同，只有说法不同** —— 这正是它存在的意义：训练只喂措辞 A、
测试给措辞 B，才能回答"策略是听懂了语义还是背下了那句话"。用 `--instructions` 可以砍掉它
（少录 1/3），代价是 **P2 的 `paraphrase` 协议没有留出措辞可用**（只剩 `combo` + `novel_object`），
生成时会打印告警。`place_left` **不能砍**：砍了意图就只剩一类，A3 的 κ 失去意义。

repo 名用 **ASCII slug**（`demo_vla_01_pick_a`）而不是中文：HF repo 名对非 ASCII 不可靠；
物体与指令的对应关系写在清单里，命令清单每条上方都有中文注释，采集时不会搞混。

用法::

    python src/vla_collect_plan.py --objects "螺丝刀,黑色长条,指甲刀" --episodes 20 \
        --out-dir runs/week38/collect

    # 砍掉 pick_b（少录 1/3；代价是 paraphrase 协议无留出措辞）：
    python src/vla_collect_plan.py --objects "螺丝刀,黑色长条,指甲刀" \
        --instructions pick_a,place_left --episodes 20 --out-dir runs/week38/collect

生成的录制命令用 `PYTHON`（`D:\miniconda\envs\lerobot\python.exe`）而不是裸 `python` —— 本机
系统 `python` 上**没有 lerobot**，裸 `python` 会 `ModuleNotFoundError: No module named 'lerobot'`。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from camera_slots import CAMERA_SLOTS  # noqa: E402  （唯一事实来源，见 camera_slots.py）

SCHEMA_VERSION = "week38_vla_collect_plan_v1"

#: ``(key, 文本模板)``。`{object}` 会被物体名替换。顺序即执行顺序，**不要随意改**（与下游协议绑定）。
DEFAULT_INSTRUCTIONS: tuple[tuple[str, str], ...] = (
    ("pick_a", "夹起桌面上的{object}"),
    ("pick_b", "把{object}拿起来"),
    ("place_left", "把{object}放到左边"),
)

#: 同义改写配对：两边都在时 P2 的 `paraphrase` 协议才有留出措辞可用；缺一边会在生成时告警。
PARAPHRASE_PARTNER_KEYS: tuple[str, str] = ("pick_a", "pick_b")

#: 相机索引：来自 `camera_slots.CAMERA_SLOTS`（唯一事实来源；索引漂移是"策略走到物体后不动作"的根因）。
CAMERA_INDICES = CAMERA_SLOTS
#: 录制时机器人/遥操作的口径（与既有采集一致）。
FOLLOWER = {"port": "COM6", "id": "my_awesome_follower_arm"}
LEADER = {"port": "COM5", "id": "my_awesome_leader_arm"}
#: 旁挂文件的存放目录。**每个 repo 一份**（见 `sidecar_for`）：`LoadSidecarWriter` 是追加写、
#: `--verify` 又把整份旁挂对**单个**数据集根目录比对，共用一个文件会让第 2 个 repo 起校验必失败。
SIDECAR_DIR = r"runs\week38\collect\sidecars"
#: 解释器：**必须用装了 lerobot 的那个**（系统 `python` 上没有 lerobot，会
#: `ModuleNotFoundError: No module named 'lerobot'`）。与既有采集 SOP 保持一致。
PYTHON = r"D:\miniconda\envs\lerobot\python.exe"
#: lerobot 本地数据集缓存根目录：数据集本地根 = `<此目录>\<repo_id>`（见 `dataset_root_for`）。
#: 这是 `--verify --dataset-root` 要填的路径（`push_to_hub=false` 时不落 Hub，只在这里）。
LEROBOT_CACHE_ROOT = r"C:\Users\Administrator\.cache\huggingface\lerobot"
#: 断开时是否卸力矩。默认 **False（不卸）**：退出时不去写 `Torque_Enable=0`，避开夹爪过载
#: 时的 `Overload error!`（实测会中断收尾；lerobot 默认 True）。代价：脚本退出后舵机**持续通电**
#: 保持姿态（好处是不下垂）。命令里显式写出该值，避免依赖 lerobot 默认值。
DISABLE_TORQUE_ON_DISCONNECT = False


class CollectPlanError(RuntimeError):
    """采集计划契约错误。"""


@dataclass(frozen=True)
class CollectTask:
    """一个采集单元 = 一个 repo = 一个物体 × 一条指令。"""

    index: int
    object_name: str
    instruction_key: str
    instruction: str
    repo_id: str
    episodes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "object": self.object_name,
            "instruction_key": self.instruction_key,
            "instruction": self.instruction,
            "repo_id": self.repo_id,
            "episodes": self.episodes,
        }


def select_instructions(
    keys: str | Sequence[str],
    instructions: Sequence[tuple[str, str]] = DEFAULT_INSTRUCTIONS,
) -> tuple[tuple[str, str], ...]:
    """按 key 从候选里挑指令（**保持 `instructions` 的顺序**，以按顺序采集）。

    `keys` 可以是逗号分隔的字符串（CLI 用）或序列。空选择、未知 key 都报错 ——
    静默当作"没选"会让生成出的清单少一整组任务而没人发现。
    """
    wanted = [item.strip() for item in (
        keys.split(",") if isinstance(keys, str) else keys
    ) if item.strip()]
    if not wanted:
        raise CollectPlanError("至少要选一条指令")
    known = {key for key, _ in instructions}
    unknown = [key for key in wanted if key not in known]
    if unknown:
        raise CollectPlanError(f"未知的指令 key {unknown}，可选：{sorted(known)}")
    selected = {key for key in wanted}
    return tuple((key, template) for key, template in instructions if key in selected)


def paraphrase_available(
    instructions: Sequence[tuple[str, str]] = DEFAULT_INSTRUCTIONS,
) -> bool:
    """同义改写配对是否齐备 —— 缺任一边，P2 的 `paraphrase` 协议就没有留出措辞可用。"""
    keys = {key for key, _ in instructions}
    return all(key in keys for key in PARAPHRASE_PARTNER_KEYS)


def build_collect_plan(
    objects: Sequence[str],
    instructions: Sequence[tuple[str, str]] = DEFAULT_INSTRUCTIONS,
    *,
    repo_prefix: str = "wangbangyan",
    dataset_prefix: str = "demo_vla",
    episodes: int = 20,
) -> tuple[CollectTask, ...]:
    """按「物体 × 指令」展开成任务清单（**保持给定顺序**，便于人按顺序采）。"""
    clean_objects = [item.strip() for item in objects if item.strip()]
    if not clean_objects:
        raise CollectPlanError("至少需要一个物体")
    if len(set(clean_objects)) != len(clean_objects):
        # 重复物体名会让两个 repo 的任务无法区分，且 combo 协议会判成"泄漏"
        raise CollectPlanError(f"物体名不能重复：{clean_objects}")
    if not instructions:
        raise CollectPlanError("至少需要一条指令模板")
    if int(episodes) <= 0:
        raise CollectPlanError(f"episodes 必须是正整数，实际 {episodes}")
    keys = [key for key, _ in instructions]
    if len(set(keys)) != len(keys):
        raise CollectPlanError(f"指令 key 不能重复：{keys}")

    tasks: list[CollectTask] = []
    for object_index, object_name in enumerate(clean_objects, start=1):
        for key, template in instructions:
            if "{object}" not in template:
                raise CollectPlanError(f"指令模板必须含 {{object}} 占位：{template!r}")
            tasks.append(
                CollectTask(
                    index=len(tasks) + 1,
                    object_name=object_name,
                    instruction_key=key,
                    instruction=template.format(object=object_name),
                    # ASCII slug：HF repo 名对非 ASCII 不可靠
                    repo_id=f"{repo_prefix}/{dataset_prefix}_{object_index:02d}_{key}",
                    episodes=int(episodes),
                )
            )
    return tuple(tasks)


def collect_combos(tasks: Sequence[CollectTask]) -> list[list[str]]:
    """转成 `[[object, instruction], ...]`，供 `run_vla_generalization --combos-json`。"""
    return [[task.object_name, task.instruction] for task in tasks]


def sidecar_for(task: CollectTask, sidecar_dir: str = SIDECAR_DIR) -> str:
    """每个 repo 一份旁挂文件（`sidecars/<repo 名>.jsonl`）——**不能共用**。

    `LoadSidecarWriter` 是追加写、`--verify` 又把整份旁挂对**单个**数据集根目录逐帧比对，
    共用一份会让第二个 repo 起校验把前一个 repo 的集号混进来 → 必然失败。
    用 `PureWindowsPath` 拼接以固定反斜杠（命令是给 PowerShell 跑的）。
    """
    name = task.repo_id.split("/")[-1] + ".jsonl"
    return str(PureWindowsPath(sidecar_dir) / name)


def dataset_root_for(task: CollectTask,
                     cache_root: str = LEROBOT_CACHE_ROOT) -> str:
    """该 repo 的**本地数据集根目录**（`--verify --dataset-root` 要填的就是它）。

    lerobot 把数据集落在 `<缓存根>\\<repo_id>`（repo_id 的 `/` 即目录分隔）。
    路径已实际确认存在：`...\\lerobot\\wangbangyan\\demo_vla_01_pick_a`。
    """
    return str(PureWindowsPath(cache_root) / PureWindowsPath(task.repo_id))


def _cameras_json() -> str:
    parts = [
        f'"{name}": {{"type": "opencv", "index_or_path": {index}, '
        f'"width": 640, "height": 480, "fps": 30}}'
        for name, index in CAMERA_INDICES.items()
    ]
    return "{" + ", ".join(parts) + "}"


def render_record_command(
    task: CollectTask,
    *,
    sidecar: str | None = None,
    runner: str = r"src\run_vla_collect_record.py",
    push_to_hub: bool = False,
    python: str = PYTHON,
    disable_torque_on_disconnect: bool = DISABLE_TORQUE_ON_DISCONNECT,
) -> str:
    """生成单条录制命令（PowerShell）。

    **必须用 `run_vla_collect_record.py` 且带 `--sidecar`**：裸 `lerobot-record` 采出的数据集
    不含负载通道，`success` 就做不了可靠标注（`demo_v23` 的教训：位置判据对人工真值错误率 57%）。
    `sidecar` 省略时按 `sidecar_for(task)` 生成**本 repo 专属**的旁挂路径。

    `python`：**必须指向装了 lerobot 的解释器**（默认 `PYTHON`），用系统 `python` 会
    `ModuleNotFoundError: No module named 'lerobot'`。

    `push_to_hub`：默认 `False`（只落本地，**先采 2 集试跑时别联网**）；铺满后再改 `--push-to-hub`
    上传私有 Hub（与既有 `demo_v23` 的私有 Hub 口径一致）。

    `disable_torque_on_disconnect`：默认 `False`（显式写入命令）—— 退出时不写 `Torque_Enable=0`，
    避开夹爪过载时收尾报 `Overload error!`（实测会中断收尾、让整份旁挂报废）。
    """
    sidecar = sidecar or sidecar_for(task)
    tail = (
        "  --dataset.push_to_hub=true --dataset.private=true --display_data=false"
        if push_to_hub
        else "  --dataset.push_to_hub=false --display_data=false"
    )
    torque = "true" if disable_torque_on_disconnect else "false"
    return "\n".join([
        f'& "{python}" {runner} `',
        f'  --sidecar {sidecar} `',
        f'  --robot.type=so101_follower --robot.port={FOLLOWER["port"]} '
        f'--robot.id={FOLLOWER["id"]} `',
        f"  --robot.cameras='{_cameras_json()}' `",
        f"  --robot.disable_torque_on_disconnect={torque} `",
        f'  --teleop.type=so101_leader --teleop.port={LEADER["port"]} '
        f'--teleop.id={LEADER["id"]} `',
        f'  --dataset.repo_id={task.repo_id} `',
        f'  --dataset.single_task="{task.instruction}" `',
        "  --dataset.num_episodes={episodes} --dataset.episode_time_s=99999 "
        "--dataset.reset_time_s=99999 `".format(episodes=task.episodes),
        tail,
    ])


def render_verify_command(
    *,
    dataset_root: str | None = None,
    sidecar: str | None = None,
    runner: str = r"src\run_vla_collect_record.py",
    python: str = PYTHON,
) -> str:
    """生成校验命令（**每条采集命令之后都要跑一次**；不通过则该份旁挂不可用）。

    `sidecar` 必须是**该 repo 专属**的旁挂文件（`sidecar_for`），否则会把别的 repo 的集号混进来。
    `dataset_root` 省略时按 `dataset_root_for(task)` 由调用方传入；直接调用本函数而不传则留占位。
    """
    sidecar = sidecar or r"<该 repo 的旁挂文件>"
    dataset_root = dataset_root or r"<该 repo 的数据集根目录>"
    return "\n".join([
        f'& "{python}" {runner} --verify `',
        f"  --dataset-root {dataset_root} `",
        f"  --sidecar {sidecar}",
    ])


def render_script(tasks: Sequence[CollectTask], *, sidecar_dir: str, runner: str,
                  push_to_hub: bool = False, python: str = PYTHON,
                  cache_root: str = LEROBOT_CACHE_ROOT,
                  disable_torque_on_disconnect: bool = DISABLE_TORQUE_ON_DISCONNECT) -> str:
    """把全部任务渲染成一个可逐条执行的 PowerShell 清单（每个 repo 一份旁挂文件）。"""
    lines = [
        "# ===== 多任务采集清单（**逐条执行**；录制是交互式的：每集按 →/Enter 结束，坏集按 ← 重录）=====",
        "# 采完每条后：跑一次 --verify，通过再进入下一个 repo。",
        "# 物体位置要**随机覆盖**左/中/右、近/远（固定轨迹会让策略只会背路径）。",
        f"# 解释器（必须装了 lerobot）：{python}",
        f"# 旁挂负载：每个 repo 一份（目录 {sidecar_dir}），由 sidecar_for() 自动生成。",
        f"# 数据集本地根：{cache_root}\\<repo 名>（--verify 的 --dataset-root 已按此填好）。",
        f"# 断开时卸力矩：{disable_torque_on_disconnect}"
        + ("（lerobot 默认）" if disable_torque_on_disconnect
           else "（不卸：避开夹爪过载时的 Overload error；退出后舵机持续通电保持姿态）"),
        f"# 上传：push_to_hub={push_to_hub}"
        + ("（私有 Hub）" if push_to_hub else "（只落本地；铺满后再用 --push-to-hub 重新生成）"),
        "",
        "# ⚠ 新建数据集要求该 repo 的本地目录**不存在**（即上面的数据集本地根）。",
        "#   中断后重跑若报 FileExistsError：先看该目录——**只有 meta/info.json 就是空壳，删掉重跑**；",
        "#   已有 episode 则改用续录（**顶层**参数，不是 dataset 子参数）：在录制命令末尾加 `--resume=true`，",
        "#   **别删**该目录（删了就丢已录数据）；续录时旁挂文件也**不要删**，会追加到同一份。",
        "",
    ]
    for task in tasks:
        sidecar = sidecar_for(task, sidecar_dir)
        lines.extend([
            f"# ---- [{task.index}/{len(tasks)}] 物体「{task.object_name}」 × {task.instruction_key}"
            f" → {task.instruction} （{task.episodes} 集）----",
            render_record_command(task, sidecar=sidecar, runner=runner,
                                  push_to_hub=push_to_hub, python=python,
                                  disable_torque_on_disconnect=disable_torque_on_disconnect),
            "# 采完校验（旁挂 vs 数据集逐帧对齐；通过再进下一个 repo）：",
            render_verify_command(dataset_root=dataset_root_for(task, cache_root),
                                  sidecar=sidecar, runner=runner, python=python),
            "",
        ])
    return "\n".join(lines)


def write_plan(tasks: Sequence[CollectTask], out_dir: str | Path,
               *, sidecar_dir: str, runner: str,
               push_to_hub: bool = False,
               python: str = PYTHON,
               cache_root: str = LEROBOT_CACHE_ROOT,
               disable_torque_on_disconnect: bool = DISABLE_TORQUE_ON_DISCONNECT,
               instructions: Sequence[tuple[str, str]] = DEFAULT_INSTRUCTIONS,
               ) -> tuple[Path, Path, Path]:
    """落盘 `collect_tasks.json` / `collect_combos.json` / `collect_commands.ps1`。"""
    directory = Path(out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    tasks_path = directory / "collect_tasks.json"
    tasks_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                # 记录**实际使用**的指令（不是全部的 DEFAULT_INSTRUCTIONS），否则清单与任务对不上
                "instructions": [{"key": k, "template": t} for k, t in instructions],
                "paraphrase_available": paraphrase_available(instructions),
                "sidecar_dir": sidecar_dir,
                "lerobot_cache_root": cache_root,
                "push_to_hub": push_to_hub,
                "python": python,
                "disable_torque_on_disconnect": disable_torque_on_disconnect,
                "tasks": [task.to_dict() for task in tasks],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    combos_path = directory / "collect_combos.json"
    combos_path.write_text(
        json.dumps(collect_combos(tasks), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    script_path = directory / "collect_commands.ps1"
    script_path.write_text(
        render_script(tasks, sidecar_dir=sidecar_dir, runner=runner,
                      push_to_hub=push_to_hub, python=python, cache_root=cache_root,
                      disable_torque_on_disconnect=disable_torque_on_disconnect),
        encoding="utf-8",
    )
    return tasks_path, combos_path, script_path


def render_table(tasks: Sequence[CollectTask]) -> str:
    lines = ["序号  repo_id                              物体        指令"]
    for task in tasks:
        lines.append(
            f"{task.index:>4}  {task.repo_id:<36}{task.object_name:<12}{task.instruction}"
        )
    total = sum(task.episodes for task in tasks)
    lines.append(
        f"合计 {len(tasks)} 个 repo / {total} 集"
        f"（约 {total * 18 / 60:.0f} 分钟遥操作，不含摆弄物体）"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="多任务采集编排（物体 × 指令 → 可执行清单）")
    parser.add_argument("--objects", default="螺丝刀,蓝色方块,白色杯子",
                        help="逗号分隔的物体名（**换成你实际的物体**）")
    parser.add_argument("--episodes", type=int, default=20, help="每个组合的集数")
    parser.add_argument("--instructions", default=",".join(k for k, _ in DEFAULT_INSTRUCTIONS),
                        help="要录的指令 key（逗号分隔）。砍掉 pick_b 可少录 1/3，"
                             "代价是 P2 的 paraphrase 协议没有留出措辞可用")
    parser.add_argument("--repo-prefix", default="wangbangyan")
    parser.add_argument("--dataset-prefix", default="demo_vla")
    parser.add_argument("--sidecar-dir", default=SIDECAR_DIR,
                        help="旁挂负载目录（**每 repo 一份**，不能共用一份文件）")
    parser.add_argument("--runner", default=r"src\run_vla_collect_record.py")
    parser.add_argument("--python", default=PYTHON,
                        help="运行 lerobot 的解释器（**必须装了 lerobot**）")
    parser.add_argument("--cache-root", default=LEROBOT_CACHE_ROOT,
                        help="lerobot 本地数据集缓存根（--verify 的 --dataset-root 由它推导）")
    parser.add_argument("--torque-off-on-disconnect", action="store_true",
                        help="退出时卸力矩（lerobot 默认；夹爪过载时会报 Overload error，"
                             "默认不卸）")
    parser.add_argument("--out-dir", default=r"runs\week38\collect")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="上传到私有 Hub（默认 false，只落本地；先试跑时别联网）")
    parser.add_argument("--print-only", action="store_true", help="只打印，不落盘")
    args = parser.parse_args()

    try:
        instructions = select_instructions(args.instructions)
        tasks = build_collect_plan(
            args.objects.split(","),
            instructions,
            episodes=args.episodes,
            repo_prefix=args.repo_prefix,
            dataset_prefix=args.dataset_prefix,
        )
    except CollectPlanError as exc:
        print(f"[采集计划] 失败：{exc}")
        return 1

    print(render_table(tasks))
    if not paraphrase_available(instructions):
        print("\n[采集计划] ⚠ 未同时包含 " + " / ".join(PARAPHRASE_PARTNER_KEYS)
              + "：P2 的 paraphrase 协议将**没有留出措辞可用**（只剩 combo + novel_object）。")
    if args.print_only:
        return 0
    tasks_path, combos_path, script_path = write_plan(
        tasks, args.out_dir, sidecar_dir=args.sidecar_dir, runner=args.runner,
        push_to_hub=args.push_to_hub, python=args.python, cache_root=args.cache_root,
        disable_torque_on_disconnect=args.torque_off_on_disconnect,
        instructions=instructions,
    )
    print(f"\n[采集计划] 清单 → {tasks_path}")
    print(f"[采集计划] 组合（喂 run_vla_generalization --combos-json）→ {combos_path}")
    print(f"[采集计划] 命令 → {script_path}（逐条执行；每条后跑一次 --verify）")
    print(f"[采集计划] 指令 → {[k for k, _ in instructions]}"
          f"（paraphrase 可用：{paraphrase_available(instructions)}）")
    print(f"[采集计划] 解释器 → {args.python}")
    print(f"[采集计划] 数据集本地根 → {args.cache_root}\\<repo 名>（--verify 已填好）")
    print(f"[采集计划] 旁挂负载 → {args.sidecar_dir}\\<repo 名>.jsonl（每 repo 一份）")
    print(f"[采集计划] 断开卸力矩：{'是' if args.torque_off_on_disconnect else '否（避开 Overload error）'}")
    print(f"[采集计划] 上传 Hub：{'是（私有）' if args.push_to_hub else '否（只落本地）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
