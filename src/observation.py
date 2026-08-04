"""将 AI2-THOR metadata 转换为 LLM 可读的结构化观察与文本摘要。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Observation:
    """当前环境状态的精简表示，仅保留对技能调用有用的信息。"""

    scene_name: str
    position: dict  # {"x": float, "y": float, "z": float}
    rotation_y: float
    held_object_ids: list[str] = field(default_factory=list)
    held_object_types: list[str] = field(default_factory=list)
    visible_pickupable: list[dict] = field(default_factory=list)
    visible_receptacles: list[dict] = field(default_factory=list)


def build_observation(metadata: dict) -> Observation:
    """从 AI2-THOR event.metadata 提取结构化 Observation。"""
    agent = metadata.get("agent", {})
    position = agent.get("position", {})
    rotation_y = agent.get("rotation", {}).get("y", 0)

    held_ids: list[str] = []
    held_types: list[str] = []
    for obj in metadata.get("inventoryObjects", []):
        held_ids.append(obj.get("objectId", ""))
        held_types.append(obj.get("objectType", ""))

    pickupable: list[dict] = []
    receptacles: list[dict] = []
    for obj in metadata.get("objects", []):
        if not obj.get("visible"):
            continue
        base = {
            "object_id": obj["objectId"],
            "object_type": obj["objectType"],
            "distance": round(obj["distance"], 2),
        }
        if obj.get("pickupable"):
            pickupable.append(base)
        if obj.get("receptacle") and obj["objectType"] != "Floor":
            receptacles.append({
                **base,
                "openable": obj.get("openable", False),
                "is_open": obj.get("isOpen", False),
            })

    pickupable.sort(key=lambda o: o["distance"])
    receptacles.sort(key=lambda o: o["distance"])

    return Observation(
        scene_name=metadata.get("sceneName", ""),
        position=position,
        rotation_y=rotation_y,
        held_object_ids=held_ids,
        held_object_types=held_types,
        visible_pickupable=pickupable,
        visible_receptacles=receptacles,
    )


def summarize(observation: Observation) -> str:
    """将 Observation 生成为简短的文本摘要，供 LLM 读取。"""
    lines: list[str] = []

    pos = observation.position
    lines.append(
        f"场景: {observation.scene_name}。"
        f"你站在 ({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})，"
        f"朝向 {observation.rotation_y:.0f}°。"
    )

    if observation.visible_pickupable:
        items = ", ".join(
            f"{o['object_type']}({o['distance']}m)" for o in observation.visible_pickupable[:10]
        )
        if len(observation.visible_pickupable) > 10:
            items += f" 等 {len(observation.visible_pickupable)} 个物体"
        lines.append(f"可见可抓取物体: {items}")
    else:
        lines.append("当前视角未发现可抓取物体。")

    if observation.visible_receptacles:
        items = []
        for r in observation.visible_receptacles[:10]:
            status = ""
            if r["openable"]:
                status = " (已打开)" if r["is_open"] else " (已关闭)"
            items.append(f"{r['object_type']}({r['distance']}m){status}")
        text = ", ".join(items)
        if len(observation.visible_receptacles) > 10:
            text += f" 等 {len(observation.visible_receptacles)} 个容器"
        lines.append(f"可见容器: {text}")
    else:
        lines.append("当前视角未发现可用容器。")

    if observation.held_object_ids:
        held = ", ".join(
            f"{t}(id={i})" for i, t in zip(observation.held_object_ids, observation.held_object_types)
        )
        lines.append(f"当前持有: {held}")
    else:
        lines.append("当前未持有物体。")

    return "\n".join(lines)
