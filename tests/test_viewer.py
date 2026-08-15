import numpy as np

from raystyle.viewer import OrbitCamera, ViewerLayout


def test_orbit_camera_interactions_remain_finite():
    camera = OrbitCamera(640, 480)
    initial = camera.pose.copy()
    camera.orbit(12, -7)
    camera.pan(4, 3)
    camera.zoom(2)
    assert camera.pose.shape == (4, 4)
    assert np.isfinite(camera.pose).all()
    assert not np.allclose(camera.pose, initial)
    assert camera.radius > 0


def test_orbit_camera_reset():
    camera = OrbitCamera(640, 480)
    camera.orbit(20, 10)
    camera.zoom(4)
    camera.reset()
    assert np.allclose(camera.center, 0)
    assert camera.radius == 2.0
    assert np.allclose(camera.rotation.as_matrix(), np.eye(3))


def test_viewer_layout_keeps_controls_spacious_for_scaled_render():
    layout = ViewerLayout(618, 411)
    assert layout.control_width >= 420
    assert layout.panel_height >= 950
    assert layout.main_width > layout.render_width
    assert layout.render_top_spacer > 100
    assert layout.viewport_width == layout.main_width + layout.panel_gap + layout.control_width


def test_viewer_layout_expands_for_large_render():
    layout = ViewerLayout(1237, 822)
    assert layout.panel_height > layout.render_height
    assert layout.render_top_spacer >= 12
