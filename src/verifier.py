"""子任务验证：基于 AI2-THOR metadata 独立验证子任务是否完成。

与 SkillResult 的区别：SkillResult 在动作执行时验证（依赖 lastActionSuccess），
SubtaskVerifier 是独立的第二层验证，直接检查 metadata 中的实际状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from task_parser import parse_instruction


@dataclass
class VerificationResult:
    """验证结果"""
    success: bool
    confidence: float  # 0.0 - 1.0
    details: str


class SubtaskVerifier:
    """通过检查 metadata 验证子任务是否实际完成。"""

    @staticmethod
    def verify_pick(
        metadata: dict[str, Any],
        object_type: str,
    ) -> VerificationResult:
        """验证物体是否真的被抓取。

        条件：inventoryObjects 中有目标类型的物体。
        """
        inventory = metadata.get("inventoryObjects", [])
        if not inventory:
            return VerificationResult(
                success=False,
                confidence=1.0,
                details="inventoryObjects 为空，未持有任何物体",
            )

        held_types = [obj.get("objectType", "") for obj in inventory]
        matched = any(
            t.casefold() == object_type.casefold() for t in held_types
        )

        if matched:
            return VerificationResult(
                success=True,
                confidence=1.0,
                details=f"inventoryObjects 中包含 {object_type}，验证通过",
            )

        held_str = ", ".join(held_types) or "无"
        return VerificationResult(
            success=False,
            confidence=1.0,
            details=f"inventoryObjects 中不含 {object_type}，当前持有: {held_str}",
        )

    @staticmethod
    def verify_place(
        metadata: dict[str, Any],
        object_id: str,
        receptacle_type: str,
    ) -> VerificationResult:
        """验证物体是否真的被放置到容器中。

        条件：目标物体的 parentReceptacles 中有目标类型容器。
        """
        objects = metadata.get("objects", [])

        # 找到目标物体
        target_obj = None
        for obj in objects:
            if obj.get("objectId") == object_id:
                target_obj = obj
                break

        if target_obj is None:
            return VerificationResult(
                success=False,
                confidence=0.5,
                details=f"在 objects 中找不到 {object_id}，无法验证放置状态",
            )

        parent_ids = target_obj.get("parentReceptacles") or []
        if not parent_ids:
            return VerificationResult(
                success=False,
                confidence=1.0,
                details=f"{object_id} 没有 parentReceptacles，不在任何容器内",
            )

        # 检查父容器是否为目标类型
        for obj in objects:
            if obj.get("objectId") in parent_ids:
                parent_type = obj.get("objectType", "")
                if parent_type.casefold() == receptacle_type.casefold():
                    return VerificationResult(
                        success=True,
                        confidence=1.0,
                        details=f"{object_id} 已在 {parent_type}({obj.get('objectId')}) 内，验证通过",
                    )

        parent_types = [
            o.get("objectType", "?")
            for o in objects
            if o.get("objectId") in parent_ids
        ]
        return VerificationResult(
            success=False,
            confidence=1.0,
            details=(
                f"{object_id} 的父容器类型为 {parent_types}，"
                f"不是预期的 {receptacle_type}"
            ),
        )

    @staticmethod
    def verify_task_complete(
        instruction: str,
        metadata: dict[str, Any],
        object_type: str | None = None,
        receptacle_type: str | None = None,
        object_id: str | None = None,
    ) -> VerificationResult:
        """精确验证目标类型物体是否位于目标类型容器中。"""
        if not object_type or not receptacle_type:
            try:
                spec = parse_instruction(instruction)
            except ValueError as exc:
                return VerificationResult(
                    success=False,
                    confidence=0.0,
                    details=f"无法解析任务目标: {exc}",
                )
            object_type = spec.object_type
            receptacle_type = spec.receptacle_type

        inventory = metadata.get("inventoryObjects", [])
        objects = metadata.get("objects", [])

        held_target_ids = {
            obj.get("objectId", "")
            for obj in inventory
            if obj.get("objectType", "").casefold() == object_type.casefold()
        }
        if object_id:
            held_target_ids &= {object_id}
        if held_target_ids:
            return VerificationResult(
                success=False,
                confidence=1.0,
                details=f"目标物体仍在手中: {sorted(held_target_ids)}",
            )

        receptacle_ids = {
            obj.get("objectId", "")
            for obj in objects
            if obj.get("objectType", "").casefold() == receptacle_type.casefold()
        }
        candidates = [
            obj
            for obj in objects
            if obj.get("objectType", "").casefold() == object_type.casefold()
            and (object_id is None or obj.get("objectId") == object_id)
        ]
        for candidate in candidates:
            matching_parents = set(candidate.get("parentReceptacles") or []) & receptacle_ids
            if matching_parents:
                return VerificationResult(
                    success=True,
                    confidence=1.0,
                    details=(
                        f"{candidate.get('objectId')} 已位于 {receptacle_type} "
                        f"{sorted(matching_parents)} 内"
                    ),
                )

        return VerificationResult(
            success=False,
            confidence=1.0,
            details=(
                f"未发现 {object_type}"
                + (f"({object_id})" if object_id else "")
                + f" 位于 {receptacle_type} 内"
            ),
        )
