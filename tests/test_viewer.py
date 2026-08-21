import numpy as np
import torch

from raystyle.viewer import (
    MethodComparisonViewerScene, OrbitCamera, RayStyleViewer, RayStyleViewerScene,
    ViewerLayout, chart_boundary_colors, chart_id_colors, uv_collision_debug_colors,
)


class _DebugField:
    resolution = 64
    chart_ids = torch.tensor([0, 1, 1, 2])
    seam_edges = torch.tensor([[0, 1], [2, 3]])
    collision_pairs = torch.tensor([[1, 2]])

    @staticmethod
    def current_local_uv():
        return torch.tensor([
            [0.0, 0.0], [0.5, 0.5], [0.505, 0.5], [1.0, 1.0],
        ])


class _DebugState:
    texture_field = _DebugField()
    selected_ids = torch.arange(4)

    @staticmethod
    def selected_albedo():
        return torch.tensor([
            [0.2, 0.3, 0.4], [0.4, 0.5, 0.6],
            [0.6, 0.7, 0.8], [0.8, 0.9, 1.0],
        ])

    @staticmethod
    def texture_preview():
        return torch.linspace(0, 1, 8).view(1, 1, 8).expand(3, 8, 8)


class _DebugBackend:
    @staticmethod
    def render_original(_camera):
        return torch.full((3, 4, 4), 0.2)

    @staticmethod
    def segment_mask(_camera, _ids):
        return torch.ones(1, 4, 4)

    @staticmethod
    def _sparse_map(_camera, _ids, values, _divisor):
        return values.mean(0).view(1, 1, 3).expand(4, 4, 3)


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


def test_viewer_exposes_all_atlas_debug_modes():
    assert {
        "Atlas albedo", "Chart boundaries", "Chart IDs",
        "Atlas texture preview", "UV collision debug",
    }.issubset(RayStyleViewer.MODES)


def test_chart_id_colors_are_stable_and_distinct():
    first = chart_id_colors(_DebugField.chart_ids)
    second = chart_id_colors(_DebugField.chart_ids)
    assert torch.equal(first, second)
    assert first.shape == (4, 3)
    assert torch.all((first >= 0) & (first <= 1))
    assert not torch.equal(first[0], first[1])


def test_chart_boundaries_are_highlighted_in_red():
    base = torch.full((4, 3), 0.6)
    colors = chart_boundary_colors(_DebugField, base)
    boundary = torch.unique(_DebugField.seam_edges.flatten())
    assert torch.all(colors[boundary, 0] == 1)
    assert torch.all(colors[boundary, 1] < 0.1)


def test_uv_collision_debug_marks_active_pairs_red():
    colors = uv_collision_debug_colors(_DebugField)
    assert torch.equal(colors[1], torch.tensor([1.0, 0.02, 0.02]))
    assert torch.equal(colors[2], torch.tensor([1.0, 0.02, 0.02]))
    assert colors[0, 2] > colors[0, 0]


def test_scene_renders_each_atlas_viewer_mode():
    scene = RayStyleViewerScene.__new__(RayStyleViewerScene)
    scene.state = _DebugState()
    scene.backend = _DebugBackend()
    camera = type("Camera", (), {"image_height": 4, "image_width": 4})()
    images = {
        mode: scene.render(camera, mode, environment=None)
        for mode in (
            "Atlas albedo", "Chart boundaries", "Chart IDs",
            "Atlas texture preview", "UV collision debug",
        )
    }
    assert all(image.shape == (3, 4, 4) for image in images.values())
    assert all(torch.isfinite(image).all() for image in images.values())
    assert not torch.equal(images["Chart IDs"], images["Chart boundaries"])
    assert not torch.equal(images["Atlas albedo"], images["UV collision debug"])


def test_method_comparison_switches_state_and_renders_every_method():
    class State:
        def __init__(self, value):
            self.value = value
            self.selected_count = 2

    class Backend:
        @staticmethod
        def render_original(_camera):
            return torch.zeros(3, 2, 2)

        @staticmethod
        def render_stylized(_camera, state, _environment, render_mode):
            assert render_mode == "pbr"
            return torch.full((3, 2, 2), state.value)

    class Train:
        albedo_only_render = False
        render_mode = "pbr"

    class Config:
        train = Train()
        method = "dummy"

    scene = MethodComparisonViewerScene.__new__(MethodComparisonViewerScene)
    scene.backend = Backend()
    scene.comparison_names = ("DC-only", "Atlas Ours")
    scene.comparison_states = {"DC-only": State(0.25), "Atlas Ours": State(0.75)}
    scene.comparison_configs = {"DC-only": Config(), "Atlas Ours": Config()}
    scene.comparison_iterations = {"DC-only": 200, "Atlas Ours": 400}
    scene._comparison_selections = {
        "DC-only": torch.tensor([True, False]),
        "Atlas Ours": torch.tensor([True, False]),
    }
    scene.states = {"Segment 1": scene.comparison_states["DC-only"]}
    scene.selections = {"Segment 1": scene._comparison_selections["DC-only"]}
    scene.segment_configs = {"Segment 1": scene.comparison_configs["DC-only"]}
    scene.segment_names = ("Segment 1",)
    scene._active_comparison = "DC-only"

    scene.set_active_comparison("Atlas Ours")
    assert scene.active_comparison_name == "Atlas Ours"
    assert scene.iteration == 400
    images = scene.render_comparison_set(camera=None, environment=None)
    assert list(images) == ["Original", "DC-only", "Atlas Ours"]
    assert torch.all(images["DC-only"] == 0.25)
    assert torch.all(images["Atlas Ours"] == 0.75)
