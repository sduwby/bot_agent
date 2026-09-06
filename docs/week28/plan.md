# 第 28 周实施计划 — 首批数据采集与质检

## 1. 任务分解

| 序号 | 子任务 | 产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 第一轮采集 demo（22 集，固定位置） | `wangbangyan/demo` | 第 27 周 SOP |
| 2 | Enter 映射与时间预算改造 | `control_utils.py` 修改 | 1 |
| 3 | 第二轮采集 demo_v2（位置随机化） | `wangbangyan/demo_v2` 51 集 | 2 |
| 4 | 质检脚本与坏集剔除 | 质检名单 | 3 |
| 5 | 第三轮定向补采 demo_v3 | `wangbangyan/demo_v3` 17 集 | 4 |
| 6 | 合并 v2+v3 并重排 | `wangbangyan/demo_v23` 68 集 | 5 |

## 2. 文件结构

### 2.1 实际新增

- 质检脚本（gripper 张开占比 / wrist 姿态统计）、合并脚本（episode 重编号、index 全局重排）。
- Hub 数据集：`wangbangyan/demo`、`wangbangyan/demo_v2`（私有）、`wangbangyan/demo_v3`、`wangbangyan/demo_v23`。
- `docs/week28/design.md` / `docs/week28/plan.md`（本文档，回溯补写）。

### 2.2 实际修改

- `control_utils.py` — 加入 Enter 键映射（结束本阶段），配合 `episode_time_s=99999`、`reset_time_s=99999` 实现按键推进的手动节奏。

## 3. 实施步骤

1. 按 SOP 采集第一轮 demo（22 集，螺丝刀固定位置）。
2. 发现模型只会背轨迹，确认必须随机化目标位置。
3. 修改 `control_utils.py` 加 Enter 映射，时间预算设 99999，手动按键控制节奏。
4. 采集 demo_v2（位置随机化，私有 Hub 仓库），质检删除 2 集坏数据：51 集 / 29506 帧。
5. 针对"接近时不张开夹爪"补采 demo_v3：原 22 集，质检删 5 集后 17 集 / 12409 帧。
6. 合并 v2+v3：任务字段 2 个 task、episode 重编号、index 全局重排，得 demo_v23（68 集 / 41915 帧）。
7. 清理未注册 episode 残留，确保 `check_timestamps_sync` 通过。

## 4. 测试计划

| 测试类型 | 方式 | 验证目标 |
| --- | --- | --- |
| 质检统计 | 逐集 gripper 张开(>12)占比、起始/末尾 wrist 姿态 | 剔除任务未完成与异常短的集 |
| 完整性 | 检查合并后数据集关键字段 | 缺失为 0 |
| 一致性 | `check_timestamps_sync` | 元数据与文件一致 |
| 可用性 | 加载 demo_v23 抽查回放 | 68 集 / 41915 帧可读 |

## 5. 验收标准

- [x] 50 条高质量成功轨迹（实际 68 集）。
- [x] 关键字段缺失为 0。

## 6. 实际偏差

1. 第一轮 demo（固定位置）整轮弃用，属计划外返工；教训写入 design.md（位置必须随机化）。
2. v3 定向补采为计划外新增轮次，由 v2 暴露的"不张开夹爪"缺陷触发。
3. 删除坏集时曾遇 `check_timestamps_sync` 校验失败，根因是未注册 episode 的残留文件；以"清理残留"作为固定操作收口。
