import torch

from raystyle.graph import AnchorGraph


def test_graph_regularizer_has_gradient():
    xyz = torch.stack((torch.arange(12).float(), torch.zeros(12), torch.zeros(12)), dim=1)
    graph = AnchorGraph.from_points(xyz, target_anchors=4, neighbours=2)
    values = torch.randn(12, 5, requires_grad=True)
    loss = graph.regularize(values)
    loss.backward()
    assert torch.isfinite(loss)
    assert values.grad is not None


def test_constant_field_has_zero_graph_loss():
    xyz = torch.randn(20, 3)
    graph = AnchorGraph.from_points(xyz, target_anchors=5, neighbours=2)
    loss = graph.regularize(torch.ones(20, 4))
    assert float(loss) < 1e-7


def test_planar_graph_keeps_nonzero_edge_weights():
    xyz = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    graph = AnchorGraph.from_points(xyz, target_anchors=4, neighbours=2)
    assert graph.edges.numel() > 0
    assert torch.all(graph.weights > 0)
    values = xyz[:, :1].clone().requires_grad_()
    loss = graph.regularize(values)
    loss.backward()
    assert float(loss.detach()) > 0
    assert torch.count_nonzero(values.grad) > 0
