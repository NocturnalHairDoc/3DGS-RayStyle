import numpy as np

from raystyle.orbit_validation import (
    CameraPose, camera_center, closed_camera_path, orbit_gates,
    ordered_camera_indices, translation_from_center,
)


def _pose(angle):
    rotation = np.eye(3, dtype=np.float32)
    center = np.array([np.cos(angle), np.sin(angle), 0.2], dtype=np.float32)
    return CameraPose(
        rotation, translation_from_center(rotation, center), 1.0, 0.8,
    )


def test_camera_center_roundtrip_matches_legacy_convention():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    center = np.array([1.5, -2.0, 0.3], dtype=np.float32)
    translation = translation_from_center(rotation, center)
    assert np.allclose(camera_center(rotation, translation), center)


def test_camera_order_is_deterministic_and_circular():
    poses = [_pose(angle) for angle in (2.0, -1.0, 0.2, -2.7)]
    first = ordered_camera_indices(poses, np.zeros(3))
    second = ordered_camera_indices(poses, np.zeros(3))
    assert first == second
    assert sorted(first) == list(range(len(poses)))


def test_closed_path_has_requested_finite_frames_and_preserves_radius():
    poses = [_pose(angle) for angle in np.linspace(-np.pi, np.pi, 8, endpoint=False)]
    path = closed_camera_path(poses, np.zeros(3), 32)
    centers = np.stack([camera_center(p.rotation, p.translation) for p in path])
    assert len(path) == 32
    assert np.isfinite(centers).all()
    assert np.allclose(np.linalg.norm(centers[:, :2], axis=1), 1.0, atol=0.08)


def test_orbit_gates_reject_temporal_jump():
    summary = {
        "minimum_correspondence_count": 256,
        "evaluated_pair_fraction": 0.9,
        "mean_corresponded_rgb_l1": 0.02,
        "p95_corresponded_rgb_l1": 0.04,
        "mean_temporal_excess_l1": 0.01,
        "p95_temporal_excess_l1": 0.02,
        "loop_closure_rgb_l1": 0.03,
        "maximum_outside_leakage": 0.002,
        "maximum_boundary_leakage": 0.01,
    }
    assert all(orbit_gates(summary).values())
    summary["p95_temporal_excess_l1"] = 0.3
    assert not orbit_gates(summary)["p95_temporal_excess_le_8pct"]
