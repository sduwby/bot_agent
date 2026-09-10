# 第 37 周 实施计划：SmolVLA 框架（S12 框架先行）— 已实现

> 状态：框架全部实现并测试（S12 框架先行）；数据/真机闭环为遗留。

## 执行记录

1. ✅ `vla_prepost.py`：`smolvla_prepare`（state + 两路图 → tensor batch）+ `smolvla_post`
   + `FakeVLAPolicy`（dry-run）；不 import 缺失的 processor_smolvla
2. ✅ `run_vla_train.py`：SmolVLA 微调训练入口（--print-only 实测通过）
3. ✅ `run_vla_infer.py`：推理 runner（真实模式为清晰 stub + 链路注释；--dry-run 实测 30 步通过）
4. ✅ `run_vla_collect.py`：多任务采集 SOP CLI
5. ✅ `tests/test_vla_prepost.py`（5 项）
6. ✅ 文档：docs/week37/design.md（含偏差回填）+ plan.md

**测试**：test_vla_prepost(5) + 既有真机相关 55 = **60 passed**；compileall 通过。

## 验收核对

1. pytest 60 项全过 ✅
2. vla_prepost 不 import processor_smolvla ✅
3. run_vla_train/collect 参数可解析、命令可生成 ✅
4. run_vla_infer --dry-run 可运行，代码路径无处理器缺失 ImportError ✅
5. design §6/§7 明确记录遗留与偏差 ✅
