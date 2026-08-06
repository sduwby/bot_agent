"""测试 SubtaskVerifier 的子任务验证功能。"""

import pytest
from verifier import SubtaskVerifier, VerificationResult


class TestVerifyPick:
    """测试 verify_pick 方法。"""

    def test_pick_success(self):
        """inventoryObjects 包含目标类型 → 验证成功。"""
        metadata = {
            "inventoryObjects": [
                {"objectId": "card_0", "objectType": "CreditCard"},
            ]
        }
        result = SubtaskVerifier.verify_pick(metadata, "CreditCard")
        assert result.success is True
        assert result.confidence == 1.0

    def test_pick_success_case_insensitive(self):
        """大小写不敏感匹配应成功。"""
        metadata = {
            "inventoryObjects": [
                {"objectId": "card_0", "objectType": "creditcard"},
            ]
        }
        result = SubtaskVerifier.verify_pick(metadata, "CreditCard")
        assert result.success is True

    def test_pick_failure_empty_inventory(self):
        """inventoryObjects 为空 → 验证失败。"""
        metadata = {"inventoryObjects": []}
        result = SubtaskVerifier.verify_pick(metadata, "CreditCard")
        assert result.success is False
        assert "为空" in result.details

    def test_pick_failure_wrong_type(self):
        """持有不同类型的物体 → 验证失败。"""
        metadata = {
            "inventoryObjects": [
                {"objectId": "apple_0", "objectType": "Apple"},
            ]
        }
        result = SubtaskVerifier.verify_pick(metadata, "CreditCard")
        assert result.success is False
        assert "Apple" in result.details

    def test_pick_failure_no_inventory_key(self):
        """metadata 缺少 inventoryObjects 键 → 验证失败。"""
        metadata = {}
        result = SubtaskVerifier.verify_pick(metadata, "CreditCard")
        assert result.success is False


class TestVerifyPlace:
    """测试 verify_place 方法。"""

    def _make_metadata(self, object_id, parent_ids, receptacle_type, receptacle_id):
        """构造包含物体和容器的 metadata。"""
        return {
            "objects": [
                {
                    "objectId": object_id,
                    "objectType": "CreditCard",
                    "parentReceptacles": parent_ids,
                },
                {
                    "objectId": receptacle_id,
                    "objectType": receptacle_type,
                },
            ]
        }

    def test_place_success(self):
        """物体在目标容器内 → 验证成功。"""
        metadata = self._make_metadata(
            object_id="card_0",
            parent_ids=["drawer_0"],
            receptacle_type="Drawer",
            receptacle_id="drawer_0",
        )
        result = SubtaskVerifier.verify_place(metadata, "card_0", "Drawer")
        assert result.success is True
        assert "验证通过" in result.details

    def test_place_failure_not_in_container(self):
        """物体不在容器内 → 验证失败。"""
        metadata = {
            "objects": [
                {"objectId": "card_0", "objectType": "CreditCard", "parentReceptacles": []},
            ]
        }
        result = SubtaskVerifier.verify_place(metadata, "card_0", "Drawer")
        assert result.success is False
        assert "没有 parentReceptacles" in result.details

    def test_place_failure_wrong_container(self):
        """物体在非目标容器内 → 验证失败。"""
        metadata = {
            "objects": [
                {"objectId": "card_0", "objectType": "CreditCard", "parentReceptacles": ["counter_0"]},
                {"objectId": "counter_0", "objectType": "CounterTop"},
            ]
        }
        result = SubtaskVerifier.verify_place(metadata, "card_0", "Drawer")
        assert result.success is False
        assert "CounterTop" in result.details

    def test_place_failure_object_not_found(self):
        """物体不在 objects 中 → 验证失败。"""
        metadata = {
            "objects": [
                {"objectId": "drawer_0", "objectType": "Drawer"},
            ]
        }
        result = SubtaskVerifier.verify_place(metadata, "card_0", "Drawer")
        assert result.success is False
        assert "找不到" in result.details


class TestVerifyTaskComplete:
    """测试 verify_task_complete 方法。"""

    def test_task_complete_object_placed(self):
        """物体已释放且在容器内 → 验证成功。"""
        metadata = {
            "inventoryObjects": [],
            "objects": [
                {"objectId": "card_0", "objectType": "CreditCard", "parentReceptacles": ["drawer_0"]},
                {"objectId": "drawer_0", "objectType": "Drawer"},
            ],
        }
        result = SubtaskVerifier.verify_task_complete("把信用卡放进抽屉", metadata)
        assert result.success is True

    def test_task_incomplete_still_holding(self):
        """仍持有物体 → 验证失败。"""
        metadata = {
            "inventoryObjects": [
                {"objectId": "card_0", "objectType": "CreditCard"},
            ],
            "objects": [],
        }
        result = SubtaskVerifier.verify_task_complete("把信用卡放进抽屉", metadata)
        assert result.success is False
        assert "仍在手中" in result.details

    def test_task_unknown_no_object_placed(self):
        """未持有物体且无容器内物体 → 不确定。"""
        metadata = {
            "inventoryObjects": [],
            "objects": [
                {"objectId": "card_0", "objectType": "CreditCard", "parentReceptacles": []},
            ],
        }
        result = SubtaskVerifier.verify_task_complete("把信用卡放进抽屉", metadata)
        assert result.success is False
        assert result.confidence == 1.0

    def test_unrelated_placed_object_does_not_complete_task(self):
        metadata = {
            "inventoryObjects": [],
            "objects": [
                {
                    "objectId": "apple_0",
                    "objectType": "Apple",
                    "parentReceptacles": ["drawer_0"],
                },
                {"objectId": "drawer_0", "objectType": "Drawer"},
            ],
        }

        result = SubtaskVerifier.verify_task_complete(
            "把信用卡放进抽屉", metadata
        )

        assert result.success is False


class TestVerificationResult:
    """测试 VerificationResult 数据类。"""

    def test_fields(self):
        """数据类字段正确。"""
        result = VerificationResult(success=True, confidence=0.9, details="ok")
        assert result.success is True
        assert result.confidence == 0.9
        assert result.details == "ok"
