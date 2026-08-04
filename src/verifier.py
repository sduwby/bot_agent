"""子任务验证：基于 AI2-THOR metadata 独立验证子任务是否完成。

与 SkillResult 的区别：SkillResult 在动作执行时验证（依赖 lastActionSuccess），
SubtaskVerifier 是独立的第二层验证，直接检查 metadata 中的实际状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    ) -> VerificationResult:
        """综合验证任务是否完成。

        对于"把 X 放进 Y"类型指令，验证 X 是否在 Y 内。
        这是一个简化实现，通过检查 inventoryObjects 和 objects 状态来判断。

        策略：
        - inventoryObjects 为空（物体已释放）
        - 存在物体的 parentReceptacles 非空（物体在某个容器内）
        """
        inventory = metadata.get("inventoryObjects", [])
        objects = metadata.get("objects", [])

        # 如果还持有物体，任务未完成
        if inventory:
            held_types = [obj.get("objectType", "") for obj in inventory]
            return VerificationResult(
                success=False,
                confidence=0.8,
                details=f"仍持有 {held_types}，任务可能未完成",
            )

        # 检查是否有物体在容器中
        placed_objects = []
        for obj in objects:
            parent_ids = obj.get("parentReceptacles") or []
            if parent_ids:
                placed_objects.append(obj.get("objectType", "?"))

        if placed_objects:
            return VerificationResult(
                success=True,
                confidence=0.8,
                details=f"检测到已放置物体: {placed_objects}，任务可能已完成",
            )

        return VerificationResult(
            success=False,
            confidence=0.5,
            details="未持有物体且未发现容器内有物体，任务状态不确定",
        )