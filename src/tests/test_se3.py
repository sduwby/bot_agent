"""SE(3) 刚体变换的纯逻辑单元测试。"""

from __future__ import annotations

from math import pi

import numpy as np
import pytest
from numpy.testing import assert_allclose

from run_transform_demo import build_demo_transforms, validate_demo
from se3 import RigidTransform, compose_chain, rotation_matrix_from_rpy


def test_identity_preserves_point_and_is_read_only() -> None:
    transform = RigidTransform.identity("world")

    assert transform.target_frame == "world"
    assert transform.source_frame == "world"
    assert_allclose(transform.transform_point([0.2, -0.3, 1.4]), [0.2, -0.3, 1.4])
    with pytest.raises(ValueError, match="read-only"):
        transform.matrix[0, 0] = 2.0


def test_pure_translation_has_known_result() -> None:
    transform = RigidTransform.from_rpy("world", "base", [1.0, -2.0, 0.5])

    assert_allclose(transform.transform_point([0.25, 0.5, -0.5]), [1.25, -1.5, 0.0])


def test_yaw_ninety_degrees_maps_x_axis_to_y_axis() -> None:
    rotation = rotation_matrix_from_rpy(0.0, 0.0, pi / 2)

    assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_rpy_uses_rz_ry_rx_order() -> None:
    roll, pitch, yaw = 0.3, -0.4, 0.7
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    rotation_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rotation_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    assert_allclose(
        rotation_matrix_from_rpy(roll, pitch, yaw),
        rotation_z @ rotation_y @ rotation_x,
        atol=1e-12,
    )


def test_compose_matches_sequential_point_transform() -> None:
    world_from_base = RigidTransform.from_rpy(
        "world", "base", [0.5, -0.1, 0.2], yaw=0.3
    )
    base_from_camera = RigidTransform.from_rpy(
        "base", "camera", [0.2, 0.0, 0.4], pitch=-0.2
    )
    point_camera = np.array([0.1, -0.2, 0.3])

    world_from_camera = world_from_base.compose(base_from_camera)
    sequential = world_from_base.transform_point(
        base_from_camera.transform_point(point_camera)
    )

    assert world_from_camera.target_frame == "world"
    assert world_from_camera.source_frame == "camera"
    assert_allclose(world_from_camera.transform_point(point_camera), sequential)


def test_inverse_restores_matrix_and_point() -> None:
    world_from_object = RigidTransform.from_rpy(
        "world",
        "object",
        [0.8, -0.2, 1.1],
        roll=0.2,
        pitch=-0.4,
        yaw=0.6,
    )
    object_from_world = world_from_object.inverse()
    point_object = np.array([0.05, -0.03, 0.12])
    point_world = world_from_object.transform_point(point_object)

    assert object_from_world.target_frame == "object"
    assert object_from_world.source_frame == "world"
    assert_allclose(object_from_world.matrix @ world_from_object.matrix, np.eye(4), atol=1e-12)
    assert_allclose(object_from_world.transform_point(point_world), point_object, atol=1e-12)


def test_compose_chain_matches_pairwise_composition() -> None:
    world_from_base = RigidTransform.from_rpy("world", "base", [1, 0, 0])
    base_from_camera = RigidTransform.from_rpy("base", "camera", [0, 2, 0])
    camera_from_object = RigidTransform.from_rpy("camera", "object", [0, 0, 3])

    chained = compose_chain(
        world_from_base,
        base_from_camera,
        camera_from_object,
    )

    assert chained.target_frame == "world"
    assert chained.source_frame == "object"
    assert_allclose(chained.transform_point([0, 0, 0]), [1, 2, 3])
    assert_allclose(
        chained.matrix,
        world_from_base.compose(base_from_camera).compose(camera_from_object).matrix,
    )


def test_constructor_copies_input_matrix() -> None:
    matrix = np.eye(4)
    matrix[0, 3] = 1.5
    transform = RigidTransform.from_matrix("world", "base", matrix)

    matrix[0, 3] = 9.0

    assert transform.matrix[0, 3] == pytest.approx(1.5)


@pytest.mark.parametrize("name", ["", "   ", None])
def test_rejects_invalid_frame_name(name: object) -> None:
    with pytest.raises(ValueError, match="非空字符串"):
        RigidTransform.from_matrix(name, "base", np.eye(4))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.eye(3), "4×4"),
        (np.full((4, 4), np.nan), "4×4"),
        (np.diag([2.0, 1.0, 1.0, 1.0]), "正交"),
        (np.diag([-1.0, 1.0, 1.0, 1.0]), "行列式"),
    ],
)
def test_rejects_invalid_se3_matrix(matrix: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RigidTransform.from_matrix("world", "base", matrix)


def test_rejects_invalid_homogeneous_last_row() -> None:
    matrix = np.eye(4)
    matrix[3, 0] = 0.1

    with pytest.raises(ValueError, match="最后一行"):
        RigidTransform.from_matrix("world", "base", matrix)


def test_rejects_non_identity_transform_with_same_frame_names() -> None:
    matrix = np.eye(4)
    matrix[0, 3] = 1.0

    with pytest.raises(ValueError, match="同名坐标系"):
        RigidTransform.from_matrix("world", "world", matrix)


@pytest.mark.parametrize("translation", [[1, 2], [1, 2, np.inf], "bad"])
def test_rejects_invalid_translation(translation: object) -> None:
    with pytest.raises(ValueError, match="translation"):
        RigidTransform.from_rpy("world", "base", translation)  # type: ignore[arg-type]


@pytest.mark.parametrize("point", [[1, 2], [1, 2, np.nan], "bad"])
def test_rejects_invalid_point(point: object) -> None:
    transform = RigidTransform.identity("world")

    with pytest.raises(ValueError, match="point"):
        transform.transform_point(point)  # type: ignore[arg-type]


def test_rejects_non_finite_rpy() -> None:
    with pytest.raises(ValueError, match="有限数值"):
        rotation_matrix_from_rpy(0.0, np.inf, 0.0)


def test_rejects_discontinuous_chain() -> None:
    world_from_base = RigidTransform.from_rpy("world", "base", [0, 0, 0])
    camera_from_object = RigidTransform.from_rpy("camera", "object", [0, 0, 0])

    with pytest.raises(ValueError, match="不连续"):
        compose_chain(world_from_base, camera_from_object)


def test_rejects_empty_chain() -> None:
    with pytest.raises(ValueError, match="至少需要一个"):
        compose_chain()


def test_demo_uses_complete_five_frame_chain() -> None:
    transforms = build_demo_transforms()

    assert [
        (transform.target_frame, transform.source_frame)
        for transform in transforms
    ] == [
        ("world", "base"),
        ("base", "end_effector"),
        ("end_effector", "camera"),
        ("camera", "object"),
    ]


def test_demo_validates_composition_and_round_trip() -> None:
    result = validate_demo()

    assert result["world_from_object"].target_frame == "world"
    assert result["world_from_object"].source_frame == "object"
    assert result["sequential_error"] < 1e-9
    assert result["round_trip_error"] < 1e-9
    assert_allclose(
        result["world_points"][0],
        [0.8779805529765273, -0.1208516429133771, 1.0704183954383992],
        atol=1e-8,
    )


@pytest.mark.parametrize("tolerance", [0.0, -1.0, np.inf])
def test_demo_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        validate_demo(tolerance=tolerance)
