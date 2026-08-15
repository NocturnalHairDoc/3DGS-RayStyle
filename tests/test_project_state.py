import json

import numpy as np

from raystyle.project_state import load_segment


def test_load_gui_segment(tmp_path):
    path = tmp_path / "state.npz"
    np.savez_compressed(path, mask=np.array([1, 2, 2, 3]), metadata=np.asarray(json.dumps({"version": 3})))
    selected, metadata = load_segment(path, segment_id=1, expected_points=4)
    assert selected.tolist() == [False, True, True, False]
    assert metadata["version"] == 3


def test_empty_segment_reports_available_ids(tmp_path):
    path = tmp_path / "state.npz"
    np.savez_compressed(path, mask=np.array([1, 2, 2]), metadata=np.asarray("{}"))
    try:
        load_segment(path, segment_id=2)
    except ValueError as error:
        assert "available" in str(error)
    else:
        raise AssertionError("empty segment was accepted")


def test_load_single_segment_pt(tmp_path):
    import torch

    path = tmp_path / "precomputed_mask.pt"
    torch.save(torch.tensor([False, True, False, True]), path)
    selected, metadata = load_segment(path, segment_id=99, expected_points=4)
    assert selected.tolist() == [False, True, False, True]
    assert metadata["format"] == "single_segment_pt"
    assert metadata["selected_gaussians"] == 2
