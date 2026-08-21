import torch

from raystyle.backend import transform_points_row


def test_legacy_projection_uses_row_vector_matrix_convention():
    points = torch.tensor([[1.0, 2.0, 3.0]])
    transform = torch.tensor([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [5.0, 6.0, 7.0, 1.0],
    ])
    assert torch.equal(
        transform_points_row(points, transform),
        torch.tensor([[7.0, 12.0, 19.0, 1.0]]),
    )
