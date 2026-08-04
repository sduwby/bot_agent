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
from replanner import Replanner
from skill_registry import get_skills_description, validate_parameters
from task_skills import SkillExecutor, SkillResult
from verifier import SubtaskVerifier


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

    def run(self, instruction: str) -> dict[str, Any]:
        """执行一次完整的 Agent 循环，返回任务摘要。"""
        self.skills.record_initial_observation()
        obs = build_observation(self.controller.last_event.metadata)
        self._observations.append(obs)
        obs_summary = summarize(obs)

        # 初始捕获：记录初始观察
        initial_memories = self.capture.extract_memories_from_observation(obs, 0)
        for m in initial_memories:
            self.store.append_memory(m)
        self.search.record_explored_position(obs.position, obs.rotation_y, 0)

        for step in range(self.max_steps):
            # 循环检测：检查是否陷入重复模式
            loop = self.monitor.detect_loop(self.action_history)
            if loop.is_loop and not self.searching:
                recovery = self.replanner.plan_recovery(loop_detection=loop)
                recovery_result = self._execute_skill(recovery.skill_name or "scan", recovery.params or {})
                obs = build_observation(self.controller.last_event.metadata)
                self._observations.append(obs)
                obs_summary = summarize(obs)
                loop_summary = f"[循环恢复] {recovery.reason}"
                self.action_history.append({
                    "skill": recovery.skill_name or "scan",
                    "parameters": recovery.params or {},
                    "success": recovery_result.success,
                    "result_summary": loop_summary,
                })
                self._record_step(step, "", loop_summary, {"skill": recovery.skill_name or "scan", "parameters": recovery.params or {}, "thought": loop.details}, recovery_result, loop_summary)
                continue

            # 搜索模式：使用 SearchStrategy 替代 LLM 决策
            if self.searching:
                search_decision = self.search.next_move()
                if search_decision["skill"] == "done":
                    return self._build_summary(instruction, False, "搜索耗尽")
                decision = search_decision
                user_prompt = build_user_prompt(
                    instruction, obs_summary, self.action_history,
                    memory_context=None,
                )
                raw_response = f"[搜索模式] {search_decision.get('thought', '')}"
            else:
                # 记忆召回：在 LLM 决策前注入相关记忆
                recall_query = self.recall.build_recall_query(
                    instruction, obs, self.action_history
                )
                recalled = self.recall.recall(recall_query, limit=5)
                memory_context = self.recall.inject_into_prompt(recalled)

                user_prompt = build_user_prompt(
                    instruction, obs_summary, self.action_history,
                    memory_context=memory_context,
                )
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

            # 更新观察
            obs = build_observation(self.controller.last_event.metadata)
            self._observations.append(obs)

            # 捕获：提取并写入 L1 记忆
            step_memories = self.capture.extract_memories_from_observation(obs, step)
            for m in step_memories:
                self.store.append_memory(m)

            if not result.success:
                # 失败时记录教训
                failure_memory = self.capture.extract_failure_memory(
                    skill_name, params, result.failure_reason or "未知原因", step, obs.scene_name
                )
                self.store.append_memory(failure_memory)

                obs_summary = summarize(obs)
                obs_summary = self._append_error_to_summary(obs_summary, result.failure_reason or "")

                # [新增] 失败分类 + 自动恢复
                failure = self.classifier.classify_failure(
                    skill_name, params, result.failure_reason or "未知"
                )
                recovery = self.replanner.plan_recovery(failure_info=failure)

                if recovery.action == "retry" and recovery.skill_name:
                    # 自动重试
                    retry_result = self._execute_skill(recovery.skill_name, recovery.params or params)
                    result_summary = f"[自动重试] {recovery.reason} → {'成功' if retry_result.success else '失败'}"
                    self.action_history.append({
                        "skill": recovery.skill_name,
                        "parameters": recovery.params or params,
                        "success": retry_result.success,
                        "result_summary": result_summary,
                    })
                    if retry_result.success:
                        self.searching = False
                        self.search.record_explored_position(obs.position, obs.rotation_y, step)
                        obs_summary = summarize(obs)
                        continue

                elif recovery.action == "scan":
                    # 自动 scan 重新观察
                    scan_result = self._execute_skill("scan", {})
                    result_summary = f"[自动恢复] {recovery.reason}"
                    self.action_history.append({
                        "skill": "scan",
                        "parameters": {},
                        "success": scan_result.success,
                        "result_summary": result_summary,
                    })
                    obs = build_observation(self.controller.last_event.metadata)
                    self._observations.append(obs)
                    obs_summary = summarize(obs)
                    continue

                elif recovery.action == "abort":
                    # 超过重试次数，交由 LLM 决策
                    obs_summary = self._append_error_to_summary(obs_summary, f"[恢复中止] {recovery.reason}")

                # 如果 pick 或 place 失败，检查是否进入搜索模式
                if not self.searching:
                    target_type = params.get("object_type") or params.get("receptacle_type")
                    if target_type and self.search.should_search_for(target_type):
                        self.searching = True
                        continue
                continue

            # 成功后退出搜索模式
            self.searching = False

            # [新增] 成功后二次验证
            if skill_name == "pick":
                verified = self.verifier.verify_pick(
                    self.controller.last_event.metadata, params.get("object_type", "")
                )
                if not verified.success:
                    obs_summary = self._append_error_to_summary(obs_summary, f"[验证警告] {verified.details}")

            elif skill_name == "place":
                held_ids = inventory_object_ids(self.controller.last_event.metadata)
                if held_ids:
                    verified = self.verifier.verify_place(
                        self.controller.last_event.metadata, held_ids[0], params.get("receptacle_type", "")
                    )
                    if not verified.success:
                        obs_summary = self._append_error_to_summary(obs_summary, f"[验证警告] {verified.details}")

            # 记录探索位置
            self.search.record_explored_position(obs.position, obs.rotation_y, step)

            obs_summary = summarize(obs)

        # 任务结束后：写入完整 L0 记录
        conv_records = self.capture.observation_to_conversation(
            instruction, self.session_key, self._observations, self.action_history
        )
        for cr in conv_records:
            self.store.append_conversation(cr)

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
