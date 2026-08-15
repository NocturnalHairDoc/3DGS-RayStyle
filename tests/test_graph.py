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

