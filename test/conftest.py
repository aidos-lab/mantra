"""Shared pytest configuration and fixtures."""

import json

import pytest
from torch_geometric.data import Data


def pytest_addoption(parser):
    parser.addoption(
        "--dataset-path",
        action="store",
        default=None,
        help="Path to manifold triangulations JSON file",
    )
    parser.addoption(
        "--dataset-path-3d",
        action="store",
        default=None,
        help="Path to 3-manifold triangulations JSON file",
    )


@pytest.fixture
def dataset_path(request):
    path = request.config.getoption("--dataset-path")
    if path is None:
        pytest.skip("No --dataset-path provided")
    return path


@pytest.fixture
def dataset(dataset_path):
    with open(dataset_path) as f:
        return json.load(f)


@pytest.fixture
def dataset_path_3d(request):
    path = request.config.getoption("--dataset-path-3d")
    if path is None:
        pytest.skip("No --dataset-path-3d provided")
    return path


@pytest.fixture
def mk_data():
    def _mk(**kw):
        d = Data()
        for k, v in kw.items():
            setattr(d, k, v)
        return d

    return _mk