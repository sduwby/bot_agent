"""Agent 主循环：观察 → LLM 规划 → 执行技能 → 记录轨迹。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from ai2thor.controller import Controller

from first_trajectory import inventory_object_ids, record_event
from llm_planner import (
    LLMBackend,
    _SYSTEM_PROMPT_TEMPLATE,
    build_user_prompt,
    create_backend,
    parse_decision,
)
from observation import Observation, build_observation, summarize
from skill_registry import get_skills_description, validate_parameters
from task_skills import SkillExecutor, SkillResult


class AgentExecutor:
    """整合主循环逻辑：观察、规划、执行、记录。"""

    def __init__(
        self,
        controller: Controller,
        run_dir,
        trajectory_file,
        records: list[dict[str, Any]],
        backend: LLMBackend | None = None,
        max_steps: int = 20,
    ) -> None:
        self.controller = controller
        self.run_dir = run_dir
        self.trajectory_file = trajectory_file
        self.records = records
        self.backend = backend or create_backend()
        self.max_steps = max_steps
        self.skills = SkillExecutor(controller, run_dir, trajectory_file, records)
        self.action_history: list[dict[str, Any]] = []
        self.system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            skills_description=get_skills_description()
        )

    def run(self, instruction: str) -> dict[str, Any]:
        """执行一次完整的 Agent 循环，返回任务摘要。"""
        self.skills.record_initial_observation()
        obs = build_observation(self.controller.last_event.metadata)
        obs_summary = summarize(obs)

        for step in range(self.max_steps):
            user_prompt = build_user_prompt(instruction, obs_summary, self.action_history)
            raw_response = self.backend.chat(self.system_prompt, user_prompt)
            decision = parse_decision(raw_response)

            if decision is None:
                self._record_step(step, user_prompt, raw_response, None, None, "决策解析失败")
                break

            skill_name = decision["skill"]
            params = decision.get("parameters", {})
            param_error = validate_parameters(skill_name, params)
            if param_error:
                self._record_step(step, user_prompt, raw_response, decision, None, param_error)
                obs_summary = self._append_error_to_summary(obs_summary, param_error)
                self.action_history.append({
                    "skill": skill_name,
                    "parameters": params,
                    "success": False,
                    "result_summary": param_error,
                })
                continue

            if skill_name == "done":
                self._record_step(step, user_prompt, raw_response, decision, None, "任务完成")
                return self._build_summary(instruction, True, "done")

            result = self._execute_skill(skill_name, params)
            result_summary = "成功" if result.success else (result.failure_reason or "失败")

            self._record_step(step, user_prompt, raw_response, decision, result, result_summary)
            self.action_history.append({
                "skill": skill_name,
                "parameters": params,
                "success": result.success,
                "result_summary": result_summary,
            })

            if not result.success:
                obs = build_observation(self.controller.last_event.metadata)
                obs_summary = summarize(obs)
                obs_summary = self._append_error_to_summary(obs_summary, result.failure_reason or "")
                continue

            obs = build_observation(self.controller.last_event.metadata)
            obs_summary = summarize(obs)

        # 达到最大步骤数
        return self._build_summary(instruction, False, "达到最大步骤数")

    def _execute_skill(self, skill_name: str, params: dict[str, Any]) -> SkillResult:
        """根据技能名称调用对应的 SkillExecutor 方法。"""
        if skill_name == "scan":
            return self._do_scan()
        if skill_name == "pick":
            return self.skills.pick(params["object_type"])
        if skill_name == "place":
            return self._do_place(params)
        if skill_name == "move_to":
            return self._do_move_to(params["direction"])
        return SkillResult(
            success=False,
            data={},
            failure_reason=f"未知技能: {skill_name}",
        )

    def _do_scan(self) -> SkillResult:
        """执行扫描并返回结构化结果。"""
        pickupable = self.skills.scan_pickupable()
        receptacles = self.skills.scan_receptacles(set())
        return SkillResult(
            success=True,
            data={
                "pickupable_count": len(pickupable),
                "receptacle_count": len(receptacles),
            },
        )

    def _do_place(self, params: dict[str, Any]) -> SkillResult:
        """执行放置，自动获取当前持有的物体 ID。"""
        held = inventory_object_ids(self.controller.last_event.metadata)
        if not held:
            return SkillResult(
                success=False,
                data={},
                failure_reason="当前未持有物体，无法放置",
            )
        object_id = held[0]
        source_parent_ids: set[str] = set()
        obj_meta = next(
            (o for o in self.controller.last_event.metadata.get("objects", [])
             if o["objectId"] == object_id),
            {},
        )
        source_parent_ids = set(obj_meta.get("parentReceptacles") or [])
        return self.skills.place(object_id, params["receptacle_type"], source_parent_ids)

    def _do_move_to(self, direction: str) -> SkillResult:
        """执行移动。"""
        action_map = {
            "forward": "MoveAhead",
            "backward": "MoveBack",
            "left": "RotateLeft",
            "right": "RotateRight",
        }
        action_name = action_map.get(direction)
        if action_name is None:
            return SkillResult(
                success=False,
                data={},
                failure_reason=f"不支持的移动方向: {direction}",
            )
        event = self.skills.act({"action": action_name})
        success = event.metadata.get("lastActionSuccess") is True
        return SkillResult(
            success=success,
            data={},
            failure_reason=event.metadata.get("errorMessage") if not success else None,
        )

    def _append_error_to_summary(self, obs_summary: str, error: str) -> str:
        """在执行失败时，将错误信息追加到观察摘要中。"""
        return f"{obs_summary}\n\n⚠ 上一步失败: {error}"

    def _record_step(
        self,
        step: int,
        user_prompt: str,
        raw_response: str,
        decision: dict[str, Any] | None,
        result: SkillResult | None,
        summary_text: str,
    ) -> None:
        """将本轮循环的关键信息写入轨迹日志。"""
        record = {
            "step": step,
            "phase": "agent_loop",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "raw_llm_response": raw_response,
            "result": asdict(result) if result else None,
            "summary": summary_text,
        }
        self.trajectory_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.trajectory_file.flush()

    def _build_summary(self, instruction: str, task_success: bool, termination_reason: str) -> dict[str, Any]:
        """构建最终的任务摘要。"""
        held = inventory_object_ids(self.controller.last_event.metadata)
        return {
            "instruction": instruction,
            "task_success": task_success,
            "termination_reason": termination_reason,
            "steps_taken": len(self.action_history),
            "action_history": self.action_history,
            "final_held_object_ids": held,
            "steps_recorded": len(self.records),
        }
