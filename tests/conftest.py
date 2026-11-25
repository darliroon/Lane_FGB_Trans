"""Shared fixtures."""
from pathlib import Path

import geopandas as gpd
import pytest

ROOT = Path(__file__).resolve().parent.parent
START = ROOT / "data" / "start.fgb"
TARGET = ROOT / "data" / "target.fgb"
V0_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "v0_result.fgb"


@pytest.fixture(scope="session")
def start():
    return START


@pytest.fixture(scope="session")
def target():
    return TARGET


@pytest.fixture(scope="session")
def origin():
    return tuple(gpd.read_file(START).total_bounds[:2])
