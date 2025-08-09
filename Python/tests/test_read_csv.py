import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_management import read_csv


@pytest.fixture
def track_csv_file(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "frame": [0, 1, 0],
            "x": [10.0, 11.0, 20.0],
            "y": [5.0, 5.5, 6.0],
            "width": [2.0, 2.0, 2.5],
            "height": [1.5, 1.5, 1.7],
            "xVelocity": [0.5, 0.6, 0.7],
            "yVelocity": [0.1, 0.1, 0.2],
            "xAcceleration": [0.01, 0.01, 0.02],
            "yAcceleration": [0.0, 0.0, 0.0],
            "frontSightDistance": [100, 101, 102],
            "backSightDistance": [50, 51, 52],
            "thw": [1.0, 1.1, 1.2],
            "ttc": [2.0, 2.1, 2.2],
            "dhw": [3.0, 3.1, 3.2],
            "precedingXVelocity": [0.4, 0.4, 0.5],
            "precedingId": [0, 0, 1],
            "followingId": [2, 2, 0],
            "leftFollowingId": [0, 0, 0],
            "leftAlongsideId": [0, 0, 0],
            "leftPrecedingId": [0, 0, 0],
            "rightFollowingId": [0, 0, 0],
            "rightAlongsideId": [0, 0, 0],
            "rightPrecedingId": [0, 0, 0],
            "laneId": [1, 1, 2],
        }
    )
    file = tmp_path / "track.csv"
    df.to_csv(file, index=False)
    return file


@pytest.fixture
def static_csv_file(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "width": [2.0, 2.5],
            "height": [1.5, 1.7],
            "initialFrame": [0, 0],
            "finalFrame": [1, 0],
            "numFrames": [2, 1],
            "class": ["car", "truck"],
            "drivingDirection": [1.0, 2.0],
            "traveledDistance": [100.0, 200.0],
            "minXVelocity": [0.4, 0.5],
            "maxXVelocity": [0.6, 0.7],
            "meanXVelocity": [0.5, 0.6],
            "minTTC": [2.0, 2.2],
            "minTHW": [1.0, 1.2],
            "minDHW": [3.0, 3.2],
            "numLaneChanges": [0, 1],
        }
    )
    file = tmp_path / "static.csv"
    df.to_csv(file, index=False)
    return file


@pytest.fixture
def meta_csv_file(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "id": [1],
            "frameRate": [25],
            "locationId": [2],
            "speedLimit": [120.0],
            "month": ["August"],
            "weekDay": ["Monday"],
            "startTime": ["08:00"],
            "duration": [3600.0],
            "totalDrivenDistance": [10000.0],
            "totalDrivenTime": [3600.0],
            "numVehicles": [2],
            "numCars": [1],
            "numTrucks": [1],
            "upperLaneMarkings": ["1;2;3"],
            "lowerLaneMarkings": ["4;5;6"],
        }
    )
    file = tmp_path / "meta.csv"
    df.to_csv(file, index=False)
    return file


def test_read_track_csv(track_csv_file: Path) -> None:
    tracks = read_csv.read_track_csv(track_csv_file)
    assert isinstance(tracks, list)
    assert len(tracks) == 2
    for track in tracks:
        assert "id" in track
        assert "bbox" in track
        assert track["bbox"].shape[0] == 4
        assert isinstance(track["frame"], np.ndarray)


def test_read_static_info(static_csv_file: Path) -> None:
    static = read_csv.read_static_info(static_csv_file)
    assert isinstance(static, dict)
    assert set(static.keys()) == {1, 2}
    for k, v in static.items():
        assert "width" in v
        assert "class" in v
        assert isinstance(v["numLaneChanges"], int)


def test_read_meta_info(meta_csv_file: Path) -> None:
    meta = read_csv.read_meta_info(meta_csv_file)
    assert isinstance(meta, dict)
    assert meta["id"] == 1
    assert meta["frameRate"] == 25
    assert isinstance(meta["upperLaneMarkings"], np.ndarray)
    assert np.allclose(meta["upperLaneMarkings"], [1, 2, 3])
