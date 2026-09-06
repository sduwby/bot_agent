"""真机记忆（阶段 F，SO-101）——S9。

简化 JSONL 记忆：任务/技能执行/失败教训/末态，供 LLM memory_context 注入。
纯逻辑：目录与文件落地，无机械臂依赖。schema 借鉴仿真版 memory_store.py。
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RealMemoryRecord:
    """一条真机记忆。"""

    content: str
    type: str = "action_experience"  # action_experience | failure_lesson | task_note
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    priority: int = 50


class RealMemoryStore:
    """JSONL 真机记忆库：追加 + 关键词检索 + LLM 文本注入。"""

    def __init__(self, base_dir: str | Path, session_key: str) -> None:
        self.base_dir = Path(base_dir)
        self.path = self.base_dir / "memories" / f"{session_key}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[RealMemoryRecord] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    data = json.loads(line)
                    self._records.append(RealMemoryRecord(**data))
                except (json.JSONDecodeError, TypeError):
                    continue

    # -- 写入 --------------------------------------------------------------
    def add(self, content: str, type_: str = "action_experience", priority: int = 50) -> RealMemoryRecord:
        rec = RealMemoryRecord(content=content, type=type_, priority=priority)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        self._records.append(rec)
        return rec

    def add_failure(self, skill_name: str, failure_reason: str, state: dict | None = None) -> RealMemoryRecord:
        detail = f" 末态={state}" if state else ""
        return self.add(
            f"技能 {skill_name} 失败: {failure_reason}{detail}",
            type_="failure_lesson", priority=70,
        )

    # -- 检索 --------------------------------------------------------------
    def search(self, query: str, type_filter: str | None = None, limit: int = 5) -> list[RealMemoryRecord]:
        terms = [t for t in query.split() if t]
        hits = []
        for rec in self._records:
            if type_filter and rec.type != type_filter:
                continue
            text = rec.content.lower()
            if any(t.lower() in text for t in terms) or not terms:
                hits.append(rec)
        hits.sort(key=lambda r: (-r.priority, r.created_at), reverse=False)
        # 按 priority 降序稳定排序
        hits.sort(key=lambda r: -r.priority)
        return hits[:limit]

    def get_summary(self) -> str:
        failures = sum(1 for r in self._records if r.type == "failure_lesson")
        return f"已记录 {len(self._records)} 条真机记忆（含 {failures} 条失败教训）"

    def inject_into_prompt(self, query: str, limit: int = 3) -> str | None:
        recs = self.search(query, limit=limit)
        if not recs:
            return None
        lines = [f"- {r.content}" for r in recs]
        return "## 场景记忆\n\n以下是本任务相关的历史记忆：\n\n" + "\n".join(lines)
