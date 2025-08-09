import sys
import pickle
from pathlib import Path
import typer
from typing import Annotated

from data_management.read_csv import read_track_csv, read_static_info, read_meta_info
from visualization.visualize_frame import VisualizationPlot


def main(
    input_path: Annotated[Path, typer.Option(help="CSV file of the tracks")] = Path(
        "../data/01_tracks.csv"
    ),
    input_static_path: Annotated[
        Path, typer.Option(help="Static meta data file for each track")
    ] = Path("../data/01_tracksMeta.csv"),
    input_meta_path: Annotated[
        Path, typer.Option(help="Static meta data file for the whole video")
    ] = Path("../data/01_recordingMeta.csv"),
    pickle_path: Annotated[
        Path,
        typer.Option(
            help="Converted pickle file that contains corresponding information of the 'input_path' file"
        ),
    ] = Path("../data/01.pickle"),
    visualize: Annotated[
        bool, typer.Option(help="True if you want to visualize the data.")
    ] = True,
    background_image: Annotated[
        Path,
        typer.Option(
            help="Optional: you can specify the correlating background image."
        ),
    ] = Path("../data/01_highway.png"),
    plotBoundingBoxes: Annotated[
        bool,
        typer.Option(
            help="Optional: decide whether to plot the bounding boxes or not."
        ),
    ] = True,
    plotDirectionTriangle: Annotated[
        bool,
        typer.Option(
            help="Optional: decide whether to plot the direction triangle or not."
        ),
    ] = True,
    plotTextAnnotation: Annotated[
        bool,
        typer.Option(
            help="Optional: decide whether to plot the text annotation or not."
        ),
    ] = True,
    plotTrackingLines: Annotated[
        bool,
        typer.Option(
            help="Optional: decide whether to plot the tracking lines or not."
        ),
    ] = True,
    plotClass: Annotated[
        bool,
        typer.Option(
            help="Optional: decide whether to show the class in the text annotation."
        ),
    ] = True,
    plotVelocity: Annotated[
        bool,
        typer.Option(
            help="Optional: decide whether to show the velocity in the text annotation."
        ),
    ] = True,
    plotIDs: Annotated[
        bool,
        typer.Option(
            help="Optional: decide whether to show the IDs in the text annotation."
        ),
    ] = True,
    save_as_pickle: Annotated[
        bool, typer.Option(help="Optional: you can save the tracks as pickle.")
    ] = False,
):
    print("Try to find the saved pickle file for better performance.")
    # Read the track csv and convert to useful format
    if pickle_path.exists():
        with pickle_path.open("rb") as fp:
            tracks = pickle.load(fp)
        print(f"Found pickle file {pickle_path}.")
    else:
        print("Pickle file not found, csv will be imported now.")
        tracks = read_track_csv(input_path)
        print("Finished importing the pickle file.")

    if save_as_pickle and not pickle_path.exists():
        print("Save tracks to pickle file.")
        with pickle_path.open("wb") as fp:
            pickle.dump(tracks, fp)

    # Read the static info
    try:
        static_info = read_static_info(input_static_path)
    except Exception as e:
        print(
            f"The static info file is either missing or contains incorrect characters. Error: {e}"
        )
        sys.exit(1)

    # Read the video meta
    try:
        meta_dictionary = read_meta_info(input_meta_path)
    except Exception as e:
        print(
            f"The video meta file is either missing or contains incorrect characters. Error: {e}"
        )
        sys.exit(1)

    if visualize:
        if tracks is None:
            print("Please specify the path to the tracks csv/pickle file.")
            sys.exit(1)
        if static_info is None:
            print("Please specify the path to the static tracks csv file.")
            sys.exit(1)
        if meta_dictionary is None:
            print("Please specify the path to the video meta csv file.")
            sys.exit(1)
        visualization_plot = VisualizationPlot(
            background_image=background_image,
            plotBoundingBoxes=plotBoundingBoxes,
            plotDirectionTriangle=plotDirectionTriangle,
            plotTextAnnotation=plotTextAnnotation,
            plotTrackingLines=plotTrackingLines,
            plotClass=plotClass,
            plotVelocity=plotVelocity,
            plotIDs=plotIDs,
            tracks=tracks,
            static_info=static_info,
            meta_dictionary=meta_dictionary,
        )
        visualization_plot.show()


if __name__ == "__main__":
    typer.run(main)
