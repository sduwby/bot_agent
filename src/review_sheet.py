"""人工抽检复核页生成（第 38 周 P1 / 子任务 2.10）。

`label_audit.compare_labels` 需要两条**独立**标注流：VLM 自动标注 + 人工抽检。
本模块把抽检样本渲染成一个自包含 HTML 页（内联 CSS/JS，无外部依赖）：

- 每集一张卡：展示抽帧 + 数据集指令（参考），人工选 intent / subtask / success；
- 页面底部按钮把结果导出为 **`LabelRecord` schema 的 jsonl**，
  可直接喂 `run_label_audit.py --human`（无需再转换）。

**页面不显示 VLM 的答案**：一旦显示，人工会照着抄，κ 被系统性抬高，
审计就失去意义——独立标注流是 A3 指标成立的前提。
"""

from __future__ import annotations

import html
import json
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vlm_labeler import LABEL_SCHEMA_VERSION, SUBTASK_NAMES, IntentTaxonomy

#: 抽检默认比例（设计要求 20%）。
DEFAULT_REVIEW_RATIO = 0.2
#: 默认随机种子，固定后抽检样本可复现。
DEFAULT_REVIEW_SEED = 38


class ReviewSheetError(RuntimeError):
    """复核页生成契约错误。"""


@dataclass(frozen=True)
class ReviewItem:
    """一条待人工复核的样本。

    ``defaults`` 为该样本各字段的**预选值**（只允许填人自己之前的答案，用于"只重问
    某一个字段"；填模型答案会造成锚定，禁止）。
    """

    episode_ref: str
    frames: tuple[str, ...]
    task: str = ""
    defaults: Mapping[str, str] = field(default_factory=dict)


def select_review_refs(
    refs: Iterable[str],
    *,
    ratio: float = DEFAULT_REVIEW_RATIO,
    count: int | None = None,
    seed: int = DEFAULT_REVIEW_SEED,
) -> list[str]:
    """按固定种子抽检 episode（可复现），返回**保持原顺序**的子集。

    `count` 给定时优先用它（便于"只抽 5 条先试"），否则取 ``round(len*ratio)``
    且至少 1 条。
    """
    ordered = list(dict.fromkeys(str(item) for item in refs))
    if not ordered:
        return []
    if count is None:
        if not 0.0 < ratio <= 1.0:
            raise ReviewSheetError(f"ratio 必须在 (0,1]，实际 {ratio}")
        count = max(1, int(round(len(ordered) * ratio)))
        if count > len(ordered):
            count = len(ordered)
    if count <= 0:
        raise ReviewSheetError("count 必须是正整数")
    count = min(count, len(ordered))
    picked = set(random.Random(seed).sample(sorted(ordered), count))
    return [ref for ref in ordered if ref in picked]


def _relative_url(path: str, base_dir: Path) -> str:
    """帧绝对路径 → 相对复核页的 URL（统一正斜杠，跨盘符时退回绝对路径）。"""
    try:
        rel = os.path.relpath(str(path), str(base_dir))
    except ValueError:  # Windows 跨盘符无法求相对路径
        rel = str(path)
    return rel.replace("\\", "/")


_OPTION_LABELS = {
    "approach": "approach（靠近）",
    "align": "align（对准）",
    "grasp": "grasp（夹住）",
    "lift": "lift（抬起）",
    "place": "place（放置）",
}


def _radios(
    group: str, options: Sequence[tuple[str, str]], selected: str | None = None
) -> str:
    """渲染一组单选按钮（``selected`` 命中时预选）。

    ``group`` 必须**逐集唯一**（形如 ``episode_000000:intent``）：HTML 中同名 radio
    是**全文档互斥**的，若各集共用 ``intent`` 这个 name，则选第 2 集的 intent 会
    自动取消第 1 集的选择 —— 表现为"整页只能填一张卡"。
    """
    cells = []
    for value, label in options:
        checked = " checked" if selected is not None and value == selected else ""
        cells.append(
            f'<label class="opt"><input type="radio" name="{html.escape(group)}" '
            f'value="{html.escape(value)}"{checked}><span>{html.escape(label)}</span></label>'
        )
    return "".join(cells)


def build_review_html(
    items: Sequence[ReviewItem],
    *,
    taxonomy: IntentTaxonomy,
    base_dir: str | Path,
    title: str = "P1 标注人工抽检",
) -> str:
    """渲染复核页 HTML（返回字符串，便于纯逻辑测试）。"""
    if not items:
        raise ReviewSheetError("没有可复核的样本")
    base = Path(base_dir).expanduser()
    intent_options = [(value, value) for value in taxonomy.intents]

    cards: list[str] = []
    for item in items:
        images = "".join(
            f'<figure><img loading="lazy" src="{html.escape(_relative_url(frame, base))}" '
            f'alt="{html.escape(Path(frame).name)}"></figure>'
            for frame in item.frames
        )
        # 逐集唯一的 radio 名字空间（见 _radios 的说明）
        prefix = item.episode_ref
        chosen = item.defaults
        cards.append(
            f"""
<section class="card" data-ref="{html.escape(item.episode_ref)}"
         data-task="{html.escape(item.task)}">
  <h2>{html.escape(item.episode_ref)}
    <span class="task">数据集指令（参考）：{html.escape(item.task) or "—"}</span></h2>
  <div class="frames">{images}</div>
  <div class="fields">
    <div class="group"><span class="label">意图 intent</span>{
        _radios(f"{prefix}:intent", intent_options, chosen.get("intent"))}</div>
    <div class="group"><span class="label">子任务 subtask</span>{
        _radios(f"{prefix}:subtask", [(n, _OPTION_LABELS[n]) for n in SUBTASK_NAMES],
                chosen.get("subtask"))}</div>
    <div class="group"><span class="label">夹爪是否夹住物体（success）</span>{
        _radios(f"{prefix}:success", [("true", "夹住了"), ("false", "没夹住")],
                chosen.get("success"))}</div>
    <div class="group"><span class="label">备注 note</span>
      <input type="text" class="note" placeholder="可选"></div>
  </div>
</section>"""
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --line:#e3e6ea; --ink:#1f2328; --muted:#6b7280; --accent:#2563eb; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font-family: "Segoe UI","Microsoft YaHei",system-ui,sans-serif; font-size:14px; }}
header {{ position:sticky; top:0; z-index:10; background:var(--card); border-bottom:1px solid var(--line);
  padding:14px 20px; }}
h1 {{ font-size:17px; margin:0 0 4px; }}
.hint {{ color:var(--muted); margin:0; font-size:12.5px; line-height:1.6; }}
main {{ padding:16px 20px 96px; max-width:1080px; margin:0 auto; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; margin-bottom:16px; }}
.card h2 {{ font-size:14px; margin:0 0 10px; font-family:ui-monospace,Consolas,monospace; }}
.card h2 .task {{ font-family:inherit; font-weight:400; color:var(--muted); margin-left:10px; }}
.frames {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }}
figure {{ margin:0; }}
.frames img {{ height:150px; border:1px solid var(--line); border-radius:6px; display:block; }}
.fields .group {{ display:flex; align-items:center; flex-wrap:wrap; gap:12px; margin:6px 0; }}
.fields .label {{ width:190px; color:var(--muted); flex:none; }}
.opt {{ display:inline-flex; align-items:center; gap:4px; cursor:pointer; }}
.opt input {{ accent-color:var(--accent); }}
.note {{ flex:1; min-width:200px; padding:4px 8px; border:1px solid var(--line); border-radius:6px; }}
footer {{ position:fixed; left:0; right:0; bottom:0; background:var(--card);
  border-top:1px solid var(--line); padding:12px 20px; display:flex; align-items:center; gap:14px; }}
button {{ background:var(--accent); color:#fff; border:0; border-radius:8px;
  padding:9px 18px; font-size:14px; cursor:pointer; }}
button:hover {{ filter:brightness(1.08); }}
#progress {{ color:var(--muted); }}
#done {{ color:#047857; font-weight:600; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <p class="hint">共 {len(items)} 集。逐集选择意图 / 子任务阶段 / 夹爪是否夹住物体（三栏都要选）。
  <b>success 的判据与自动标注一致：只看<b>夹爪是否夹住了物体</b>（不问是否抬起）</b>——
  两侧口径必须相同，否则一致率无从比较。
  页面<b>不显示</b>模型答案——请独立判断，导出后与自动标注比对。填完后点右下角导出，得到
  <code>human.jsonl</code>，再运行
  <code>python src/run_label_audit.py --vlm &lt;自动标注目录&gt; --human human.jsonl</code>。</p>
</header>
<main>{"".join(cards)}</main>
<footer>
  <button id="export">导出 human.jsonl</button>
  <span id="progress"></span><span id="done"></span>
</footer>
<script>
const TAXONOMY_VERSION = {json.dumps(taxonomy.version)};
const SCHEMA_VERSION = {json.dumps(LABEL_SCHEMA_VERSION)};
function cards() {{ return Array.from(document.querySelectorAll('.card')); }}
// radio 的 name 是「<episode_ref>:<字段>」逐集唯一的，故用「结尾匹配」在卡片内取，
// 避免依赖 episode_ref 的转义；也防止跨集串组（同名 radio 全文档互斥）。
function picked(card, name) {{
  const el = card.querySelector(`input[name$=":${{name}}"]:checked`);
  return el ? el.value : null;
}}
function collect(skipIncomplete) {{
  const rows = [];
  for (const card of cards()) {{
    const intent = picked(card, 'intent');
    const subtask = picked(card, 'subtask');
    const success = picked(card, 'success');
    if (intent === null || subtask === null || success === null) {{
      if (skipIncomplete) continue;
      return {{ rows: null, missing: card.dataset.ref }};
    }}
    rows.push({{
      schema_version: SCHEMA_VERSION,
      episode_ref: card.dataset.ref,
      task_text: card.dataset.task || '',
      frame_range: null,
      intent: intent,
      subtask: subtask,
      success: success === 'true',
      confidence: 1.0,
      source: 'human',
      taxonomy_version: TAXONOMY_VERSION,
      note: (card.querySelector('.note').value || '').trim()
    }});
  }}
  return {{ rows: rows, missing: null }};
}}
function refresh() {{
  const total = cards().length;
  let done = 0;
  for (const card of cards()) {{
    if (picked(card, 'intent') && picked(card, 'subtask') && picked(card, 'success')) done++;
  }}
  document.getElementById('progress').textContent = `已填 ${{done}} / ${{total}}`;
}}
document.addEventListener('change', refresh);
document.getElementById('export').addEventListener('click', () => {{
  const result = collect(false);
  if (result.rows === null) {{
    alert('还有未填完的样本：' + result.missing + '（三栏都要选）');
    return;
  }}
  const text = result.rows.map(r => JSON.stringify(r)).join('\\n') + '\\n';
  const blob = new Blob([text], {{ type: 'application/jsonl' }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'human.jsonl';
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  document.getElementById('done').textContent =
    `已导出 ${{result.rows.length}} 条 → 请把 human.jsonl 放回自动标注目录后运行审计`;
}});
refresh();
</script>
</body>
</html>
"""


def write_review_sheet(
    path: str | Path,
    items: Sequence[ReviewItem],
    *,
    taxonomy: IntentTaxonomy,
    title: str = "P1 标注人工抽检",
) -> Path:
    """把复核页写到 ``path``（图片按相对路径引用，放同目录/子目录即可）。"""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_review_html(items, taxonomy=taxonomy, base_dir=target.parent, title=title),
        encoding="utf-8",
    )
    return target
