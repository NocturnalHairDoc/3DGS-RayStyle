import torch

from raystyle.reference_layout import (
    build_reference_layout, metric_tile_grid, reference_saliency,
)


def _structured_reference():
    image = torch.full((3, 80, 120), 0.2)
    image[0, 12:44, 68:104] = 0.95
    image[1, 20:36, 76:96] = 0.8
    image[2, 24:32, 80:92] = 0.05
    return image


def test_saliency_grid_is_deterministic_across_random_seeds():
    reference = _structured_reference()
    torch.manual_seed(1)
    first = build_reference_layout(reference, "saliency_grid", 4, 96)
    torch.manual_seed(9876)
    second = build_reference_layout(reference, "saliency_grid", 4, 96)
    assert first.regions == second.regions
    assert first.scales == second.scales
    assert torch.equal(first.canvas, second.canvas)
    assert torch.equal(first.saliency, second.saliency)


def test_saliency_layout_is_finite_bounded_and_preserves_colour_statistics():
    reference = _structured_reference()
    layout = build_reference_layout(reference, "saliency_grid", 4, 128)
    assert layout.canvas.shape == (3, 128, 128)
    assert torch.isfinite(layout.canvas).all()
    assert torch.all((layout.canvas >= 0) & (layout.canvas <= 1))
    assert torch.allclose(
        layout.canvas.mean((1, 2)), reference.mean((1, 2)), atol=0.035,
    )
    for x0, y0, x1, y1 in layout.regions:
        assert 0 <= x0 < x1 <= 1
        assert 0 <= y0 < y1 <= 1


def test_saliency_layout_handles_flat_and_tiny_references():
    reference = torch.full((3, 5, 7), 0.42)
    layout = build_reference_layout(reference, "saliency_grid", 4, 32)
    assert torch.isfinite(layout.canvas).all()
    assert torch.allclose(layout.canvas, torch.full_like(layout.canvas, 0.42), atol=1e-5)
    assert torch.isfinite(reference_saliency(reference)).all()


def test_full_layout_keeps_original_reference_exactly():
    reference = torch.rand(3, 19, 27)
    layout = build_reference_layout(reference, "full", 4, 64)
    assert torch.equal(layout.canvas, reference)
    assert layout.regions == ((0.0, 0.0, 1.0, 1.0),)


def test_saliency_grid_enlarges_the_structured_region():
    reference = _structured_reference()
    layout = build_reference_layout(reference, "saliency_grid", 4, 128)
    def gradient_energy(image):
        dx = (image[..., 1:] - image[..., :-1]).abs().mean()
        dy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean()
        return dx + dy

    assert gradient_energy(layout.canvas) > gradient_energy(reference) * 1.25


def test_saliency_focus_uses_one_bounded_structured_crop():
    reference = _structured_reference()
    layout = build_reference_layout(
        reference, "saliency_focus", canvas_size=96, focus_scale=0.3,
    )
    assert layout.canvas.shape == (3, 96, 96)
    assert layout.scales == (0.3,)
    assert len(layout.regions) == 1
    assert torch.isfinite(layout.canvas).all()


def test_saliency_tile_repeats_the_focus_crop_deterministically():
    reference = _structured_reference()
    layout = build_reference_layout(
        reference, "saliency_tile", canvas_size=96,
        focus_scale=0.3, tile_count=3,
    )
    assert layout.canvas.shape == (3, 96, 96)
    first = layout.canvas[:, :32, :32]
    assert torch.allclose(first, layout.canvas[:, 32:64, 32:64], atol=1e-6)


def test_saliency_motifs_selects_distinct_deterministic_regions():
    reference = torch.full((3, 120, 120), 0.15)
    for y, x, colour in (
        (12, 14, (0.95, 0.75, 0.1)),
        (14, 76, (0.1, 0.8, 0.95)),
        (72, 20, (0.9, 0.15, 0.2)),
        (74, 78, (0.2, 0.9, 0.25)),
    ):
        reference[:, y:y + 24, x:x + 24] = reference.new_tensor(colour)[:, None, None]
        reference[:, y + 7:y + 17, x + 7:x + 17] = 0.02
    torch.manual_seed(3)
    first = build_reference_layout(
        reference, "saliency_motifs", patch_count=4, canvas_size=96,
        focus_scale=0.22, tile_grid=(4, 1),
    )
    torch.manual_seed(900)
    second = build_reference_layout(
        reference, "saliency_motifs", patch_count=4, canvas_size=96,
        focus_scale=0.22, tile_grid=(4, 1),
    )
    assert first.regions == second.regions
    assert torch.equal(first.canvas, second.canvas)
    assert len(first.regions) == 4
    assert first.tile_grid == (4, 1)
    for index, region in enumerate(first.regions):
        for other in first.regions[index + 1:]:
            assert _box_iou(region, other) <= 0.31


def _box_iou(first, second):
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    a = (first[2] - first[0]) * (first[3] - first[1])
    b = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(a + b - intersection, 1e-8)


def test_metric_tile_grid_compensates_long_planar_axis():
    class Field:
        component_ids = torch.zeros(60, dtype=torch.long)
        chart_ids = torch.zeros(60, dtype=torch.long)
        chart_axes = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])

    x, y = torch.meshgrid(
        torch.linspace(0, 4, 10), torch.linspace(0, 1, 6), indexing="xy",
    )
    xyz = torch.stack((x.flatten(), y.flatten(), torch.zeros(60)), dim=1)
    columns, rows = metric_tile_grid(Field(), xyz, base_count=3)
    assert columns >= 11
    assert rows == 3
