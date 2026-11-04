"""Shared fixtures: a small deterministic synthetic world (warehouse + trained model + RAG index) built once per test session."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from p2p.common.config import reset_settings_cache


@pytest.fixture(scope="session")
def world(tmp_path_factory) -> Path:
    data = tmp_path_factory.mktemp("p2p-data")
    os.environ["DATA_DIR"] = str(data)
    reset_settings_cache()
    from p2p.data.synthetic import build_world, write_world
    from p2p.ml.train import train
    from p2p.rag.index import HybridIndex

    wh = write_world(build_world(seed=5, n_vendors=110), data)
    train(wh.path, data / "models")
    HybridIndex.build(wh.documents()).save(data / "rag")
    # module-level caches must not leak between sessions
    import p2p.rag.index as rag
    rag._cached = None
    return data


@pytest.fixture(scope="session")
def wh(world):
    from p2p.data.repository import Warehouse
    return Warehouse(world / "warehouse.sqlite", readonly=True)


@pytest.fixture(scope="session")
def labels(world):
    from p2p.data.repository import LabelStore
    return LabelStore(world / "warehouse.sqlite", readonly=True).labels()


@pytest.fixture(scope="session")
def by_type(labels):
    out: dict[str, list[str]] = {}
    for iid, (_, kind) in labels.items():
        out.setdefault(kind, []).append(iid)
    return out


@pytest.fixture(scope="session")
def engine(world):
    from p2p.services.scoring.engine import ScoringEngine
    return ScoringEngine()
