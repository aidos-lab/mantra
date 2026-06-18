"""Tests for ``mantra.datasets.pairwise``."""

import pytest

import mantra.datasets.pairwise as pairwise_mod
from mantra.datasets import MANTRA
from mantra.datasets.pairwise import PairwiseSimplicialDS
from mantra.manifold_types import Manifold2Type
from mantra.transforms.create_labels import CreateLabels


@pytest.fixture
def patch_mantra(monkeypatch, make_manifolds_json, pairwise_entries):
    """Patch the inner MANTRA to load a local fixture (no network).

    The fixture mixes three classes: ``S^2`` and ``T^2`` (the comparison
    pair) plus ``RP^2`` (which must be excluded from the pairing).
    """
    path = make_manifolds_json(pairwise_entries)

    def fake_mantra(root, **kwargs):
        kwargs.setdefault("local_path", path)
        return MANTRA(root, **kwargs)

    monkeypatch.setattr(pairwise_mod, "MANTRA", fake_mantra)
    return path


def test_pairwise_construction_and_filtering(patch_mantra, tmp_path):
    ds = PairwiseSimplicialDS(
        str(tmp_path / "root"),
        comparison_pair=(Manifold2Type.S_2, Manifold2Type.T_2),
        # A tensor attribute (``y``) so the ``torch.cat`` branch is taken
        # alongside the tuple branch for the non-tensor attributes.
        pre_transform=CreateLabels("orientable"),
    )
    # 2 spheres + 2 tori = 4 matching objects -> 4 * 3 ordered pairs that
    # both fall inside the comparison pair (RP^2 is excluded).
    assert len(ds) == 12
    pair = ds[0]
    # Each attribute now stores a 2-element comparison.
    assert pair.name == ("S^2", "S^2") or len(pair.name) == 2
    assert pair.y.shape[0] == 2


def test_raw_file_names_and_download(patch_mantra, tmp_path):
    ds = PairwiseSimplicialDS(
        str(tmp_path / "root"),
        comparison_pair=(Manifold2Type.S_2, Manifold2Type.T_2),
        pre_transform=CreateLabels("orientable"),
    )
    assert ds.raw_file_names == []
    assert ds.download() is None
    assert (
        ds._data_filename(Manifold2Type.S_2, Manifold2Type.T_2)
        == "data_s_2_t_2.pt"
    )


def test_unknown_pair_raises_value_error(patch_mantra, tmp_path):
    ds = PairwiseSimplicialDS(
        str(tmp_path / "root"),
        comparison_pair=(Manifold2Type.S_2, Manifold2Type.T_2),
        pre_transform=CreateLabels("orientable"),
    )

    class _Fake:
        name = "NOT_A_REAL_PAIR"

    with pytest.raises(ValueError, match="Can not find"):
        ds._get_processed_path(_Fake(), _Fake())


def test_dimension_3_not_implemented(patch_mantra, tmp_path):
    with pytest.raises(NotImplementedError):
        PairwiseSimplicialDS(
            str(tmp_path / "root"),
            comparison_pair=(Manifold2Type.S_2, Manifold2Type.T_2),
            dimension=3,
            pre_transform=CreateLabels("orientable"),
        )
