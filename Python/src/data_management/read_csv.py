import pandas as pd
import numpy as np
from typing import Any
from pathlib import Path


# TRACK FILE
BBOX = "bbox"
FRAMES = "frames"
FRAME = "frame"
TRACK_ID = "id"
X = "x"
Y = "y"
WIDTH = "width"
HEIGHT = "height"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"
FRONT_SIGHT_DISTANCE = "frontSightDistance"
BACK_SIGHT_DISTANCE = "backSightDistance"
DHW = "dhw"
THW = "thw"
TTC = "ttc"
PRECEDING_X_VELOCITY = "precedingXVelocity"
PRECEDING_ID = "precedingId"
FOLLOWING_ID = "followingId"
LEFT_PRECEDING_ID = "leftPrecedingId"
LEFT_ALONGSIDE_ID = "leftAlongsideId"
LEFT_FOLLOWING_ID = "leftFollowingId"
RIGHT_PRECEDING_ID = "rightPrecedingId"
RIGHT_ALONGSIDE_ID = "rightAlongsideId"
RIGHT_FOLLOWING_ID = "rightFollowingId"
LANE_ID = "laneId"

# STATIC FILE
INITIAL_FRAME = "initialFrame"
FINAL_FRAME = "finalFrame"
NUM_FRAMES = "numFrames"
CLASS = "class"
DRIVING_DIRECTION = "drivingDirection"
TRAVELED_DISTANCE = "traveledDistance"
MIN_X_VELOCITY = "minXVelocity"
MAX_X_VELOCITY = "maxXVelocity"
MEAN_X_VELOCITY = "meanXVelocity"
MIN_DHW = "minDHW"
MIN_THW = "minTHW"
MIN_TTC = "minTTC"
NUMBER_LANE_CHANGES = "numLaneChanges"

# VIDEO META
ID = "id"
FRAME_RATE = "frameRate"
LOCATION_ID = "locationId"
SPEED_LIMIT = "speedLimit"
MONTH = "month"
WEEKDAY = "weekDay"
START_TIME = "startTime"
DURATION = "duration"
TOTAL_DRIVEN_DISTANCE = "totalDrivenDistance"
TOTAL_DRIVEN_TIME = "totalDrivenTime"
N_VEHICLES = "numVehicles"
N_CARS = "numCars"
N_TRUCKS = "numTrucks"
UPPER_LANE_MARKINGS = "upperLaneMarkings"
LOWER_LANE_MARKINGS = "lowerLaneMarkings"


def read_track_csv(input_path: str | Path) -> list[dict[str, Any]]:
    """
    Reads the tracks file from highD data and returns a list of track dictionaries.

    Args:
        input_path (str or Path): Path to the tracks CSV file. The file must contain columns: id, frame, x, y, width, height, xVelocity, yVelocity, xAcceleration, yAcceleration, frontSightDistance, backSightDistance, thw, ttc, dhw, precedingXVelocity, precedingId, followingId, leftFollowingId, leftAlongsideId, leftPrecedingId, rightFollowingId, rightAlongsideId, rightPrecedingId, laneId.

    Returns:
        list[dict[str, Any]]: A list containing all tracks as dictionaries.

    Raises:
        RuntimeError: If reading or processing the CSV file fails (e.g., file not found, missing columns, or invalid format).
    """
    try:
        df: pd.DataFrame = pd.read_csv(input_path)
        required_columns = [
            TRACK_ID,
            FRAME,
            X,
            Y,
            WIDTH,
            HEIGHT,
            X_VELOCITY,
            Y_VELOCITY,
            X_ACCELERATION,
            Y_ACCELERATION,
            FRONT_SIGHT_DISTANCE,
            BACK_SIGHT_DISTANCE,
            THW,
            TTC,
            DHW,
            PRECEDING_X_VELOCITY,
            PRECEDING_ID,
            FOLLOWING_ID,
            LEFT_FOLLOWING_ID,
            LEFT_ALONGSIDE_ID,
            LEFT_PRECEDING_ID,
            RIGHT_FOLLOWING_ID,
            RIGHT_ALONGSIDE_ID,
            RIGHT_PRECEDING_ID,
            LANE_ID,
        ]
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Missing required columns in track CSV file '{input_path}': {missing}"
            )
        grouped = df.groupby([TRACK_ID], sort=False)
        tracks: list[dict[str, Any]] = [None] * grouped.ngroups
        current_track: int = 0
        for group_id, rows in grouped:
            bounding_boxes = np.transpose(
                np.array(
                    [
                        rows[X].values,
                        rows[Y].values,
                        rows[WIDTH].values,
                        rows[HEIGHT].values,
                    ]
                )
            )
            tracks[current_track] = {
                TRACK_ID: np.int64(group_id),
                FRAME: rows[FRAME].values,
                BBOX: bounding_boxes,
                X_VELOCITY: rows[X_VELOCITY].values,
                Y_VELOCITY: rows[Y_VELOCITY].values,
                X_ACCELERATION: rows[X_ACCELERATION].values,
                Y_ACCELERATION: rows[Y_ACCELERATION].values,
                FRONT_SIGHT_DISTANCE: rows[FRONT_SIGHT_DISTANCE].values,
                BACK_SIGHT_DISTANCE: rows[BACK_SIGHT_DISTANCE].values,
                THW: rows[THW].values,
                TTC: rows[TTC].values,
                DHW: rows[DHW].values,
                PRECEDING_X_VELOCITY: rows[PRECEDING_X_VELOCITY].values,
                PRECEDING_ID: rows[PRECEDING_ID].values,
                FOLLOWING_ID: rows[FOLLOWING_ID].values,
                LEFT_FOLLOWING_ID: rows[LEFT_FOLLOWING_ID].values,
                LEFT_ALONGSIDE_ID: rows[LEFT_ALONGSIDE_ID].values,
                LEFT_PRECEDING_ID: rows[LEFT_PRECEDING_ID].values,
                RIGHT_FOLLOWING_ID: rows[RIGHT_FOLLOWING_ID].values,
                RIGHT_ALONGSIDE_ID: rows[RIGHT_ALONGSIDE_ID].values,
                RIGHT_PRECEDING_ID: rows[RIGHT_PRECEDING_ID].values,
                LANE_ID: rows[LANE_ID].values,
            }
            current_track += 1
        return tracks
    except Exception as e:
        raise RuntimeError(
            f"Error reading or processing the track CSV file '{input_path}': {e}"
        )


def read_static_info(input_static_path: str | Path) -> dict[int, dict[str, Any]]:
    """
    Reads the static info file from highD data and returns a dictionary with track_id as key.

    Args:
        input_static_path (str or Path): Path to the static CSV file. The file must contain columns: id, width, height, initialFrame, finalFrame, numFrames, class, drivingDirection, traveledDistance, minXVelocity, maxXVelocity, meanXVelocity, minTTC, minTHW, minDHW, numLaneChanges.

    Returns:
        dict[int, dict[str, Any]]: The static dictionary - the key is the track_id and the value is the corresponding data for this track.

    Raises:
        RuntimeError: If reading or processing the CSV file fails (e.g., file not found, missing columns, or invalid format).
    """
    try:
        df: pd.DataFrame = pd.read_csv(input_static_path)
        required_columns = [
            TRACK_ID,
            WIDTH,
            HEIGHT,
            INITIAL_FRAME,
            FINAL_FRAME,
            NUM_FRAMES,
            CLASS,
            DRIVING_DIRECTION,
            TRAVELED_DISTANCE,
            MIN_X_VELOCITY,
            MAX_X_VELOCITY,
            MEAN_X_VELOCITY,
            MIN_TTC,
            MIN_THW,
            MIN_DHW,
            NUMBER_LANE_CHANGES,
        ]
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Missing required columns in static CSV file '{input_static_path}': {missing}"
            )
    except Exception as e:
        raise RuntimeError(
            f"Error reading the static CSV file '{input_static_path}': {e}"
        )
    try:
        # Optimize type conversion for all relevant columns at once
        df = df.astype(
            {
                TRACK_ID: int,
                WIDTH: float,
                HEIGHT: float,
                INITIAL_FRAME: int,
                FINAL_FRAME: int,
                NUM_FRAMES: int,
                CLASS: str,
                DRIVING_DIRECTION: float,
                TRAVELED_DISTANCE: float,
                MIN_X_VELOCITY: float,
                MAX_X_VELOCITY: float,
                MEAN_X_VELOCITY: float,
                MIN_TTC: float,
                MIN_THW: float,
                MIN_DHW: float,
                NUMBER_LANE_CHANGES: int,
            }
        )
        static_dict: dict[int, dict[str, Any]] = {}
        for row in df.itertuples(index=False):
            track_id: int = getattr(row, TRACK_ID)
            static_dict[track_id] = {
                TRACK_ID: track_id,
                WIDTH: getattr(row, WIDTH),
                HEIGHT: getattr(row, HEIGHT),
                INITIAL_FRAME: getattr(row, INITIAL_FRAME),
                FINAL_FRAME: getattr(row, FINAL_FRAME),
                NUM_FRAMES: getattr(row, NUM_FRAMES),
                CLASS: getattr(row, CLASS),
                DRIVING_DIRECTION: getattr(row, DRIVING_DIRECTION),
                TRAVELED_DISTANCE: getattr(row, TRAVELED_DISTANCE),
                MIN_X_VELOCITY: getattr(row, MIN_X_VELOCITY),
                MAX_X_VELOCITY: getattr(row, MAX_X_VELOCITY),
                MEAN_X_VELOCITY: getattr(row, MEAN_X_VELOCITY),
                MIN_TTC: getattr(row, MIN_TTC),
                MIN_THW: getattr(row, MIN_THW),
                MIN_DHW: getattr(row, MIN_DHW),
                NUMBER_LANE_CHANGES: getattr(row, NUMBER_LANE_CHANGES),
            }
        return static_dict
    except Exception as e:
        raise RuntimeError(
            f"Error processing the static CSV file '{input_static_path}': {e}"
        )


def read_meta_info(input_meta_path: str | Path) -> dict[str, Any]:
    """
    Reads the video meta file from highD data and returns a dictionary with general video information.

    Args:
        input_meta_path (str or Path): Path to the video meta CSV file. The file must contain columns: id, frameRate, locationId, speedLimit, month, weekDay, startTime, duration, totalDrivenDistance, totalDrivenTime, numVehicles, numCars, numTrucks, upperLaneMarkings, lowerLaneMarkings.

    Returns:
        dict[str, Any]: The meta dictionary containing the general information of the video.

    Raises:
        RuntimeError: If reading or processing the CSV file fails (e.g., file not found, missing columns, or invalid format).
    """
    try:
        df: pd.DataFrame = pd.read_csv(input_meta_path)
        required_columns = [
            ID,
            FRAME_RATE,
            LOCATION_ID,
            SPEED_LIMIT,
            MONTH,
            WEEKDAY,
            START_TIME,
            DURATION,
            TOTAL_DRIVEN_DISTANCE,
            TOTAL_DRIVEN_TIME,
            N_VEHICLES,
            N_CARS,
            N_TRUCKS,
            UPPER_LANE_MARKINGS,
            LOWER_LANE_MARKINGS,
        ]
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Missing required columns in meta CSV file '{input_meta_path}': {missing}"
            )
    except Exception as e:
        raise RuntimeError(f"Error reading the meta CSV file '{input_meta_path}': {e}")
    try:
        meta_dict: dict[str, Any] = {
            ID: int(df[ID][0]),
            FRAME_RATE: int(df[FRAME_RATE][0]),
            LOCATION_ID: int(df[LOCATION_ID][0]),
            SPEED_LIMIT: float(df[SPEED_LIMIT][0]),
            MONTH: str(df[MONTH][0]),
            WEEKDAY: str(df[WEEKDAY][0]),
            START_TIME: str(df[START_TIME][0]),
            DURATION: float(df[DURATION][0]),
            TOTAL_DRIVEN_DISTANCE: float(df[TOTAL_DRIVEN_DISTANCE][0]),
            TOTAL_DRIVEN_TIME: float(df[TOTAL_DRIVEN_TIME][0]),
            N_VEHICLES: int(df[N_VEHICLES][0]),
            N_CARS: int(df[N_CARS][0]),
            N_TRUCKS: int(df[N_TRUCKS][0]),
            UPPER_LANE_MARKINGS: np.fromstring(df[UPPER_LANE_MARKINGS][0], sep=";"),
            LOWER_LANE_MARKINGS: np.fromstring(df[LOWER_LANE_MARKINGS][0], sep=";"),
        }
        return meta_dict
    except Exception as e:
        raise RuntimeError(
            f"Error processing the meta CSV file '{input_meta_path}': {e}"
        )
