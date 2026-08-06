"""Agent 主循环：观察 → LLM 规划 → 执行技能 → 记录轨迹。"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from ai2thor.controller import Controller

from action_monitor import ActionMonitor
from failure_classifier import FailureClassifier
from first_trajectory import inventory_object_ids, record_event
from llm_planner import (
    LLMBackend,
    _SYSTEM_PROMPT_TEMPLATE,
    build_user_prompt,
    create_backend,
    parse_decision,
)
from memory_capture import MemoryCapture
from memory_recall import MemoryRecall
from memory_store import MemoryStore, SearchStrategy
from observation import Observation, build_observation, summarize
from replanner import RecoveryAction, Replanner
from skill_registry import get_skills_description, validate_parameters
from task_parser import TaskSpec, parse_instruction
from task_skills import SkillExecutor, SkillResult
from verifier import SubtaskVerifier, VerificationResult


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
        max_search_steps: int = 8,
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

        # 记忆模块
        self.session_key = uuid.uuid4().hex[:8]
        self.store = MemoryStore(run_dir, self.session_key)
        self.capture = MemoryCapture()
        self.recall = MemoryRecall(self.store)
        self.search = SearchStrategy(self.store, max_search_steps)
        self.searching = False
        self._observations: list[Observation] = []

        # 验证与恢复模块
        self.monitor = ActionMonitor()
        self.classifier = FailureClassifier()
        self.verifier = SubtaskVerifier()
        self.replanner = Replanner()
        self._pending_recovery: dict[str, Any] | None = None
        self._search_target_type: str | None = None
        self._search_probe_pending = False
        self._target_object_id: str | None = None
        self._task_spec: TaskSpec | None = None
        self._final_verification: VerificationResult | None = None
        self._l0_written = False

    def run(self, instruction: str) -> dict[str, Any]:
        """执行一次完整的 Agent 循环，返回任务摘要。"""
        try:
            try:
                self._task_spec = parse_instruction(instruction)
            except ValueError as exc:
                return self._build_summary(
                    instruction, False, f"任务解析失败: {exc}"
                )

            self.skills.record_initial_observation()
            obs = build_observation(self.controller.last_event.metadata)
            self._observations.append(obs)
            obs_summary = summarize(obs)
            self._capture_observation(obs, 0)
            self.search.record_explored_position(obs.position, obs.rotation_y, 0)

            while len(self.action_history) < self.max_steps:
                step = len(self.action_history)
                decision: dict[str, Any] | None = None
                user_prompt = build_user_prompt(
                    instruction, obs_summary, self.action_history
                )
                raw_response = ""

                if self._pending_recovery is not None:
                    decision = self._pending_recovery
                    self._pending_recovery = None
                    raw_response = f"[自动恢复] {decision.get('thought', '')}"

                elif self.searching:
                    if self._target_visible(obs, self._search_target_type):
                        self.searching = False
                    elif self._search_probe_pending:
                        decision = self._build_search_probe_decision()
                        self._search_probe_pending = False
                        raw_response = (
                            f"[搜索探测] {decision.get('thought', '')}"
                        )
                    else:
                        decision = self.search.next_move(obs.position, obs.rotation_y)
                        if decision["skill"] == "done":
                            return self._build_summary(
                                instruction, False, "搜索耗尽"
                            )
                        raw_response = f"[搜索模式] {decision.get('thought', '')}"

                if decision is None:
                    loop = self.monitor.detect_loop(self.action_history)
                    if loop.is_loop:
                        recovery = self.replanner.plan_recovery(
                            loop_detection=loop
                        )
                        decision = {
                            "thought": recovery.reason,
                            "skill": recovery.skill_name or "scan",
                            "parameters": recovery.params or {},
                        }
                        raw_response = f"[循环恢复] {loop.details}"

                if decision is None:
                    recall_query = self.recall.build_recall_query(
                        instruction, obs, self.action_history
                    )
                    recalled = self.recall.recall(recall_query, limit=5)
                    memory_context = self.recall.inject_into_prompt(recalled)
                    user_prompt = build_user_prompt(
                        instruction,
                        obs_summary,
                        self.action_history,
                        memory_context=memory_context,
                    )
                    raw_response = self.backend.chat(
                        self.system_prompt, user_prompt
                    )
                    decision = parse_decision(raw_response)

                if decision is None:
                    self._record_step(
                        step,
                        user_prompt,
                        raw_response,
                        None,
                        None,
                        "决策解析失败",
                    )
                    return self._build_summary(
                        instruction, False, "决策解析失败"
                    )

                skill_name = decision["skill"]
                params = decision.get("parameters", {})
                param_error = validate_parameters(skill_name, params)
                if param_error:
                    self._record_step(
                        step,
                        user_prompt,
                        raw_response,
                        decision,
                        None,
                        param_error,
                    )
                    self.action_history.append({
                        "skill": skill_name,
                        "parameters": params,
                        "success": False,
                        "result_summary": param_error,
                    })
                    obs_summary = self._append_error_to_summary(
                        summarize(obs), param_error
                    )
                    continue

                if skill_name == "done":
                    self._final_verification = self._verify_task(instruction)
                    details = self._final_verification.details
                    self._record_step(
                        step,
                        user_prompt,
                        raw_response,
                        decision,
                        None,
                        details,
                    )
                    if self._final_verification.success:
                        return self._build_summary(
                            instruction, True, "任务状态验证通过"
                        )
                    self.action_history.append({
                        "skill": "done",
                        "parameters": {},
                        "success": False,
                        "result_summary": f"任务验证失败: {details}",
                    })
                    obs_summary = self._append_error_to_summary(
                        summarize(obs), f"任务验证失败: {details}"
                    )
                    continue

                placed_object_id = None
                if skill_name == "place":
                    held_ids = inventory_object_ids(
                        self.controller.last_event.metadata
                    )
                    placed_object_id = held_ids[0] if held_ids else None

                result = self._execute_skill(skill_name, params)
                obs = build_observation(self.controller.last_event.metadata)
                verification = self._verify_skill_result(
                    skill_name, params, result, placed_object_id
                )
                if verification is not None and not verification.success:
                    result = SkillResult(
                        success=False,
                        data={
                            **result.data,
                            "verification": asdict(verification),
                        },
                        failure_reason=f"验证失败: {verification.details}",
                    )

                if result.success and skill_name == "pick":
                    target = result.data.get("target", {})
                    self._target_object_id = target.get("object_id")
                    if not self._target_object_id:
                        held = inventory_object_ids(
                            self.controller.last_event.metadata
                        )
                        self._target_object_id = held[0] if held else None

                result_summary = (
                    "成功"
                    if result.success
                    else (result.failure_reason or "失败")
                )
                if verification is not None:
                    result_summary += f"；验证: {verification.details}"

                self.action_history.append({
                    "skill": skill_name,
                    "parameters": params,
                    "success": result.success,
                    "result_summary": result_summary,
                })
                self._record_step(
                    step,
                    user_prompt,
                    raw_response,
                    decision,
                    result,
                    result_summary,
                )
                self._observations.append(obs)
                self._capture_observation(obs, step + 1)
                self.search.record_explored_position(
                    obs.position, obs.rotation_y, step + 1
                )
                obs_summary = summarize(obs)

                if result.success:
                    self.replanner.record_success(skill_name)
                    if self.searching and skill_name == "move_to":
                        self._search_probe_pending = True
                    if self._target_visible(obs, self._search_target_type):
                        self.searching = False
                    if skill_name in {"pick", "place"}:
                        self.searching = False
                        self._search_probe_pending = False
                        self._search_target_type = None
                        self.search.reset(clear_explored=False)
                    continue

                failure_reason = result.failure_reason or "未知原因"
                self.store.append_memory(
                    self.capture.extract_failure_memory(
                        skill_name,
                        params,
                        failure_reason,
                        step + 1,
                        obs.scene_name,
                    )
                )
                obs_summary = self._append_error_to_summary(
                    obs_summary, failure_reason
                )
                failure = self.classifier.classify_failure(
                    skill_name, params, failure_reason
                )
                recovery = self.replanner.plan_recovery(
                    failure_info=failure
                )
                self._apply_recovery(recovery, params)

            self._final_verification = self._verify_task(instruction)
            if self._final_verification.success:
                return self._build_summary(
                    instruction, True, "达到步骤上限时任务状态验证通过"
                )
            return self._build_summary(
                instruction, False, "达到最大步骤数"
            )
        finally:
            self._persist_l0(instruction)

    def _capture_observation(self, obs: Observation, step: int) -> None:
        """把最新观察完整写入 L1。"""
        for memory in self.capture.extract_memories_from_observation(obs, step):
            self.store.append_memory(memory)

    def _verify_skill_result(
        self,
        skill_name: str,
        params: dict[str, Any],
        result: SkillResult,
        placed_object_id: str | None,
    ) -> VerificationResult | None:
        """在技能报告成功后使用最新 metadata 做独立状态验证。"""
        if not result.success:
            return None
        metadata = self.controller.last_event.metadata
        if skill_name == "pick":
            return self.verifier.verify_pick(
                metadata, params.get("object_type", "")
            )
        if skill_name == "place":
            if not placed_object_id:
                return VerificationResult(
                    success=False,
                    confidence=1.0,
                    details="放置前未记录到持有物体 ID",
                )
            return self.verifier.verify_place(
                metadata,
                placed_object_id,
                params.get("receptacle_type", ""),
            )
        return None

    def _verify_task(self, instruction: str) -> VerificationResult:
        """按解析后的任务目标验证最终环境状态。"""
        if self._task_spec is None:
            return VerificationResult(
                success=False,
                confidence=0.0,
                details="任务目标尚未解析",
            )
        return self.verifier.verify_task_complete(
            instruction,
            self.controller.last_event.metadata,
            object_type=self._task_spec.object_type,
            receptacle_type=self._task_spec.receptacle_type,
            object_id=self._target_object_id,
        )

    @staticmethod
    def _target_visible(
        obs: Observation,
        target_type: str | None,
    ) -> bool:
        """判断当前视野是否已出现正在搜索的物体或容器类型。"""
        if not target_type:
            return False
        target = target_type.casefold()
        visible_types = {
            item["object_type"].casefold()
            for item in obs.visible_pickupable + obs.visible_receptacles
        }
        return target in visible_types

    def _apply_recovery(
        self,
        recovery: RecoveryAction,
        failed_params: dict[str, Any],
    ) -> None:
        """把恢复计划转换为下一轮统一执行的决策或搜索状态。"""
        if recovery.action == "search":
            target_type = (
                failed_params.get("object_type")
                or failed_params.get("receptacle_type")
            )
            if target_type and target_type != self._search_target_type:
                self.search.reset(clear_explored=False)
            self._search_target_type = target_type
            self.searching = bool(target_type)
            self._search_probe_pending = False
            return
        if recovery.action in {"retry", "scan"} and recovery.skill_name:
            self._pending_recovery = {
                "thought": recovery.reason,
                "skill": recovery.skill_name,
                "parameters": recovery.params or failed_params,
            }

    def _build_search_probe_decision(self) -> dict[str, Any]:
        """移动后用目标技能的 360 度扫描确认是否已经到达可交互范围。"""
        if (
            self._task_spec is not None
            and self._search_target_type == self._task_spec.receptacle_type
        ):
            return {
                "thought": f"全向探测容器 {self._search_target_type}",
                "skill": "place",
                "parameters": {
                    "receptacle_type": self._search_target_type,
                },
            }
        return {
            "thought": f"全向探测物体 {self._search_target_type}",
            "skill": "pick",
            "parameters": {"object_type": self._search_target_type},
        }

    def _persist_l0(self, instruction: str) -> None:
        """在任意终止路径写入一次完整 L0 对话。"""
        if self._l0_written:
            return
        records = self.capture.observation_to_conversation(
            instruction,
            self.session_key,
            self._observations,
            self.action_history,
        )
        for record in records:
            self.store.append_conversation(record)
        self._l0_written = True

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
            "task": asdict(self._task_spec) if self._task_spec else None,
            "task_success": task_success,
            "termination_reason": termination_reason,
            "final_verification": (
                asdict(self._final_verification)
                if self._final_verification
                else None
            ),
            "steps_taken": len(self.action_history),
            "action_history": self.action_history,
            "final_held_object_ids": held,
            "steps_recorded": len(self.records),
        }
