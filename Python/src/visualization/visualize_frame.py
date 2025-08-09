import os
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from matplotlib.widgets import Button, Slider
from typing import Any, Dict, List, Optional
from data_management.read_csv import (
    TRACK_ID,
    FINAL_FRAME,
    INITIAL_FRAME,
    X_VELOCITY,
    BBOX,
    UPPER_LANE_MARKINGS,
    LOWER_LANE_MARKINGS,
)

mpl.rcParams["savefig.dpi"] = 300


class VisualizationPlot:
    def update_figure(self) -> None:
        """Draw all bounding boxes, direction triangles, text annotations, and tracking lines for the current frame."""
        rect_style = dict(facecolor="r", fill=True, edgecolor="k", zorder=19)
        triangle_style = dict(
            facecolor="k", fill=True, edgecolor="k", lw=0.1, alpha=0.6, zorder=19
        )
        text_style = dict(picker=True, size=8, color="k", zorder=10, ha="center")
        text_box_style = dict(
            boxstyle="round,pad=0.2", fc="yellow", alpha=0.6, ec="black", lw=0.2
        )
        track_style = dict(color="r", linewidth=1, zorder=10)

        plotted_objects = []
        for track in self.tracks:
            track_id = track[TRACK_ID][0]
            static_track_information = self.static_info[track_id]
            initial_frame = static_track_information[INITIAL_FRAME]
            final_frame = static_track_information[FINAL_FRAME]
            if initial_frame <= self.current_frame < final_frame:
                current_index = self.current_frame - initial_frame
                try:
                    bounding_box = np.array(track[BBOX][current_index])
                    if self.arguments.get("background_image") is not None:
                        bounding_box = bounding_box / 0.10106 / 4
                    y_position = self.y_sign * bounding_box[1]
                except (IndexError, KeyError, ValueError) as e:
                    print(f"Error accessing bounding box: {e}")
                    continue

                bounding_box_y = y_position
                # Plot the rectangle of the bounding box
                if self.arguments.get("plotBoundingBoxes", True):
                    if self.y_sign < 0:
                        bounding_box_y += self.y_sign * bounding_box[3]
                    rect = plt.Rectangle(
                        (bounding_box[0], bounding_box_y),
                        bounding_box[2],
                        bounding_box[3],
                        **rect_style,
                    )
                    self.ax.add_patch(rect)
                    plotted_objects.append(rect)

                # Plot direction triangle
                if self.arguments.get("plotDirectionTriangle", True):
                    current_velocity = track[X_VELOCITY][current_index]
                    triangle_y_position = [
                        y_position,
                        y_position + bounding_box[3],
                        y_position + (bounding_box[3] / 2),
                    ]
                    if self.y_sign < 0:
                        triangle_y_position = [
                            y + self.y_sign * bounding_box[3]
                            for y in triangle_y_position
                        ]
                        bounding_box_y += self.y_sign * bounding_box[3]
                    if current_velocity < 0:
                        x_back_position = bounding_box[0] + (bounding_box[2] * 0.2)
                        triangle_info = np.array(
                            [
                                [x_back_position, x_back_position, bounding_box[0]],
                                triangle_y_position,
                            ]
                        )
                    else:
                        x_back_position = (
                            bounding_box[0] + bounding_box[2] - (bounding_box[2] * 0.2)
                        )
                        triangle_info = np.array(
                            [
                                [
                                    x_back_position,
                                    x_back_position,
                                    bounding_box[0] + bounding_box[2],
                                ],
                                triangle_y_position,
                            ]
                        )
                    polygon = plt.Polygon(np.transpose(triangle_info), **triangle_style)
                    self.ax.add_patch(polygon)
                    plotted_objects.append(polygon)

                # Plot text annotation
                if self.arguments.get("plotTextAnnotation", True):
                    vehicle_class = (
                        static_track_information.get("class", [None])[0]
                        if "class" in static_track_information
                        else None
                    )
                    if vehicle_class is None and "CLASS" in globals():
                        vehicle_class = static_track_information["CLASS"][0]
                    annotation_text = ""
                    if (
                        self.arguments.get("plotClass", True)
                        and vehicle_class is not None
                    ):
                        annotation_text += f"{vehicle_class}"
                    if self.arguments.get("plotVelocity", True):
                        if annotation_text != "":
                            annotation_text += "|"
                        x_velocity = abs(float(track[X_VELOCITY][current_index])) * 3.6
                        annotation_text += f"{x_velocity:.2f}km/h"
                    if self.arguments.get("plotIDs", True):
                        if annotation_text != "":
                            annotation_text += "|"
                        annotation_text += f"ID{track_id}"
                    if self.background_image is not None:
                        target_location = (bounding_box[0], y_position - 1)
                        text_location = (
                            bounding_box[0] + (bounding_box[2] / 2),
                            y_position - 1.5,
                        )
                    else:
                        target_location = (bounding_box[0], y_position + 1)
                        text_location = (
                            bounding_box[0] + (bounding_box[2] / 2),
                            y_position + 1.5,
                        )
                    text_patch = self.ax.annotate(
                        annotation_text,
                        xy=target_location,
                        xytext=text_location,
                        bbox=text_box_style,
                        **text_style,
                    )
                    plotted_objects.append(text_patch)

                # Plot tracking line for each vehicle
                if self.arguments.get("plotTrackingLines", True):
                    relevant_bounding_boxes = np.array(track[BBOX][0:current_index, :])
                    if relevant_bounding_boxes.shape[0] > 0:
                        if self.arguments.get("background_image") is not None:
                            relevant_bounding_boxes = (
                                relevant_bounding_boxes / 0.10106 / 4
                            )
                        sign = 1 if self.background_image is not None else self.y_sign
                        x_centroid_position = (
                            relevant_bounding_boxes[:, 0]
                            + relevant_bounding_boxes[:, 2] / 2
                        )
                        y_centroid_position = (
                            sign * relevant_bounding_boxes[:, 1]
                        ) + sign * (relevant_bounding_boxes[:, 3]) / 2
                        centroids = np.column_stack(
                            (x_centroid_position, y_centroid_position)
                        )
                        track_sign = 1 if track[X_VELOCITY][current_index] < 0 else -1
                        plotted_centroids = self.ax.plot(
                            centroids[:, 0] + track_sign * (bounding_box[3] / 2),
                            centroids[:, 1],
                            **track_style,
                        )
                        plotted_objects.append(plotted_centroids)

        self.fig.canvas.mpl_connect("pick_event", self.on_click)
        self.plotted_objects = plotted_objects

    """
    Provides interactive visualization for highD dataset tracks.

    Args:
        arguments (dict): Visualization settings and file paths.
        read_tracks (List[Dict[str, Any]]): List of track dictionaries.
        static_info (Dict[int, Dict[str, Any]]): Static info for each track.
        meta_dictionary (Dict[str, Any]): Meta information for the video.
        fig (Optional[plt.Figure]): Optional matplotlib figure to use.
    """

    # Class-level constants for configuration
    FIG_SIZE = (32, 4)
    SLIDER_COLOR = "lightgoldenrodyellow"
    OUTER_LINE_THICKNESS = 0.2
    LANE_COLOR = "white"

    def __init__(
        self,
        background_image: Optional[str] = None,
        plotBoundingBoxes: bool = True,
        plotDirectionTriangle: bool = True,
        plotTextAnnotation: bool = True,
        plotTrackingLines: bool = True,
        plotClass: bool = True,
        plotVelocity: bool = True,
        plotIDs: bool = True,
        tracks: Optional[List[Dict[str, Any]]] = None,
        static_info: Optional[Dict[int, Dict[str, Any]]] = None,
        meta_dictionary: Optional[Dict[str, Any]] = None,
        fig: Optional[plt.Figure] = None,
    ) -> None:
        """Initialize the VisualizationPlot object with visualization options and data."""
        # Store arguments in a dictionary for internal use
        self.arguments: Dict[str, Any] = {
            "background_image": background_image,
            "plotBoundingBoxes": plotBoundingBoxes,
            "plotDirectionTriangle": plotDirectionTriangle,
            "plotTextAnnotation": plotTextAnnotation,
            "plotTrackingLines": plotTrackingLines,
            "plotClass": plotClass,
            "plotVelocity": plotVelocity,
            "plotIDs": plotIDs,
        }
        self.tracks: List[Dict[str, Any]] = tracks if tracks is not None else []
        self.static_info: Dict[int, Dict[str, Any]] = (
            static_info if static_info is not None else {}
        )
        self.meta_dictionary: Dict[str, Any] = (
            meta_dictionary if meta_dictionary is not None else {}
        )
        self.current_frame: int = 1
        self.changed_button: bool = False
        self.rect_map: Dict = {}
        self.plotted_objects: List = []

        # Calculate maximum frames
        if self.tracks:
            last_track = self.tracks[-1]
            self.maximum_frames: int = (
                self.static_info[last_track[TRACK_ID][0]][FINAL_FRAME] - 1
            )
        else:
            self.maximum_frames: int = 1

        # Create figure and axes
        if fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
            self.fig.set_size_inches(*self.FIG_SIZE)
            plt.subplots_adjust(left=0.0, right=1.0, bottom=0.20, top=1.0)
        else:
            self.fig = fig
            self.ax = self.fig.gca()

        # Set up background
        self._setup_background()

        # Initialize the plot with the bounding boxes of the first frame
        self.update_figure()

        # Set up widgets
        self._setup_widgets()

        self.ax.set_autoscale_on(False)

    def _setup_background(self) -> None:
        """Set up the background image or virtual lanes."""
        background_image_path = self.arguments.get("background_image")
        if background_image_path and os.path.exists(background_image_path):
            self.background_image = imread(background_image_path)
            self.y_sign = 1
            im = self.background_image[:, :, :]
            self.ax.imshow(im)
        else:
            self.background_image = None
            self.y_sign = -1
            self.outer_line_thickness = self.OUTER_LINE_THICKNESS
            self.lane_color = self.LANE_COLOR
            self.plot_highway()

    def _setup_widgets(self) -> None:
        """Set up the interactive widgets (slider and buttons)."""
        self.ax_slider = self.fig.add_axes(
            [0.1, 0.05, 0.2, 0.03], facecolor=self.SLIDER_COLOR
        )
        self.ax_button_previous2 = self.fig.add_axes([0.02, 0.035, 0.026, 0.07])
        self.ax_button_previous = self.fig.add_axes([0.05, 0.035, 0.02, 0.07])
        self.ax_button_next = self.fig.add_axes([0.325, 0.035, 0.02, 0.07])
        self.ax_button_next2 = self.fig.add_axes([0.35, 0.035, 0.025, 0.07])

        self.frame_slider = Slider(
            self.ax_slider,
            "Frame",
            1,
            self.maximum_frames,
            valinit=self.current_frame,
            valfmt="%s",
            valstep=1,
        )
        self.button_previous2 = Button(self.ax_button_previous2, "Previous x5")
        self.button_previous = Button(self.ax_button_previous, "Previous")
        self.button_next = Button(self.ax_button_next, "Next")
        self.button_next2 = Button(self.ax_button_next2, "Next x5")

        self.frame_slider.on_changed(self.update_slider)
        self.button_previous.on_clicked(self.update_button_previous)
        self.button_next.on_clicked(self.update_button_next)
        self.button_previous2.on_clicked(self.update_button_previous2)
        self.button_next2.on_clicked(self.update_button_next2)

    def update_slider(self, value: float) -> None:
        """
        Callback for the frame slider. Updates the current frame and refreshes the plot.
        Args:
            value (float): The new frame value from the slider.
        """
        if not self.changed_button:
            self.current_frame = int(value)
            self.remove_patches()
            self.update_figure()
            self.fig.canvas.draw_idle()
        self.changed_button = False

    def update_button_next(self, _event: Any) -> None:
        """Callback for the 'Next' button."""
        if self.current_frame < self.maximum_frames:
            self.current_frame += 1
            self.changed_button = True
            self.trigger_update()

    def trigger_update(self) -> None:
        """Update the plot after a button press."""
        self.remove_patches()
        self.update_figure()
        self.frame_slider.set_val(self.current_frame)
        self.fig.canvas.draw_idle()

    def plot_highway(self) -> None:
        """Draw the highway lanes and markings."""
        upper_lanes = self.meta_dictionary[UPPER_LANE_MARKINGS]
        lower_lanes = self.meta_dictionary[LOWER_LANE_MARKINGS]
        # Top lane area
        self.ax.add_patch(
            plt.Rectangle(
                (0, self.y_sign * lower_lanes[-1] - 5),
                400,
                lower_lanes[-1] - upper_lanes[0] + 10,
                color="grey",
                fill=True,
                alpha=1,
                zorder=5,
            )
        )
        self.ax.add_patch(
            plt.Rectangle(
                (0, self.y_sign * upper_lanes[0]),
                400,
                self.outer_line_thickness,
                color=self.lane_color,
                fill=True,
                alpha=1,
                zorder=5,
            )
        )
        for i in range(1, len(upper_lanes) - 1):
            plt.plot(
                (0, 400),
                (self.y_sign * upper_lanes[i], self.y_sign * upper_lanes[i]),
                color=self.lane_color,
                linestyle="dashed",
                dashes=(25, 70),
                alpha=1,
                zorder=5,
            )
        self.ax.add_patch(
            plt.Rectangle(
                (0, self.y_sign * upper_lanes[-1]),
                400,
                self.outer_line_thickness,
                color=self.lane_color,
                fill=True,
                alpha=1,
                zorder=5,
            )
        )
        # Bottom lane area
        self.ax.add_patch(
            plt.Rectangle(
                (0, self.y_sign * lower_lanes[0]),
                400,
                self.outer_line_thickness,
                color=self.lane_color,
                alpha=1,
                zorder=5,
            )
        )
        for i in range(1, len(lower_lanes) - 1):
            plt.plot(
                (0, 400),
                (self.y_sign * lower_lanes[i], self.y_sign * lower_lanes[i]),
                color=self.lane_color,
                linestyle="dashed",
                dashes=(25, 70),
                alpha=1,
                zorder=5,
            )
        self.ax.add_patch(
            plt.Rectangle(
                (0, self.y_sign * lower_lanes[-1]),
                400,
                self.outer_line_thickness,
                color=self.lane_color,
                alpha=1,
                zorder=5,
            )
        )

    def on_click(self, event):
        artist = event.artist
        text_value = artist._text
        try:
            track_id = int(text_value[text_value.find("ID") + 2 :])
            selected_track = None
            for track in self.tracks:
                if track[TRACK_ID] == track_id:
                    selected_track = track
                    break
            if selected_track is None:
                print(
                    "No track with the ID {} was found. Nothing to show.".format(
                        track_id
                    )
                )
                return
            static_information = self.static_info[track_id]
            # Get bounding boxes for the selected track
            bounding_box = selected_track[BBOX]
            centroids = [
                bounding_box[:, 0] + bounding_box[:, 2] / 2,
                bounding_box[:, 1] + bounding_box[:, 3] / 2,
            ]
            centroids = np.transpose(centroids)
            initial_frame = static_information[INITIAL_FRAME]
            final_frame = static_information[FINAL_FRAME]
            x_limits = [initial_frame, final_frame]
            track_frames = np.linspace(
                initial_frame, final_frame, centroids.shape[0], dtype=np.int64
            )
            # Create a new figure that pops up
            fig = plt.figure(np.random.randint(0, 5000, 1))
            fig.canvas.set_window_title("Track {}".format(track_id))
            plt.suptitle("Information for track {}.".format(track_id))
            # Plot the x positions
            plt.subplot(311, title="X-Position")
            x_positions = centroids[:, 0]
            borders = [np.amin(x_positions), np.amax(x_positions)]
            plt.plot(track_frames, x_positions)
            plt.plot([self.current_frame, self.current_frame], borders, "--r")
            plt.xlim(x_limits)
            offset = (borders[1] - borders[0]) * 0.05
            borders = [borders[0] - offset, borders[1] + offset]
            plt.ylim(borders)
            plt.xlabel("Frame")
            plt.ylabel("X-Position [m]")

            # Plot the y positions
            plt.subplot(312, title="Y-Position")
            y_positions = centroids[:, 1]
            borders = [np.amin(y_positions), np.amax(y_positions)]
            plt.plot(track_frames, y_positions)
            plt.plot([self.current_frame, self.current_frame], borders, "--r")
            plt.xlim(x_limits)
            offset = (borders[1] - borders[0]) * 0.05
            borders = [borders[0] - offset, borders[1] + offset]
            plt.ylim(borders)
            plt.xlabel("Frame")
            plt.ylabel("Y-Position [m]")

            # Plot the velocity
            plt.subplot(313, title="X-Velocity")
            velocity = abs(selected_track[X_VELOCITY])
            borders = [np.amin(velocity), np.amax(velocity)]
            plt.plot(track_frames, velocity)
            plt.plot([self.current_frame, self.current_frame], borders, "--r")
            plt.xlim(x_limits)
            offset = (borders[1] - borders[0]) * 0.05
            borders = [borders[0] - offset, borders[1] + offset]
            plt.ylim(borders)
            plt.xlabel("Frame")
            plt.ylabel("X-Velocity [m/s]")

            plt.subplots_adjust(wspace=0.1, hspace=1)
            plt.show()
        except Exception as e:
            print(
                f"Something went wrong when trying to plot the detailed information of the vehicle: {e}"
            )
            return

    def get_figure(self) -> plt.Figure:
        """Return the matplotlib figure object."""
        return self.fig

    def remove_patches(self) -> None:
        """Remove all plotted objects from the axes."""
        self.fig.canvas.mpl_disconnect("pick_event")
        for figure_object in self.plotted_objects:
            if isinstance(figure_object, list):
                figure_object[0].remove()
            else:
                figure_object.remove()

    @staticmethod
    def show() -> None:
        """Show and close the matplotlib plot."""
        plt.show()
        plt.close()
