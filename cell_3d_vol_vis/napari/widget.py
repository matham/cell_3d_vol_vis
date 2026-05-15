""" """

import random
from functools import partial
from pathlib import Path
from typing import Literal

import matplotlib.colors as mcolors
import napari
import napari.layers
import numpy as np
from brainglobe_utils.cells.cells import Cell
from cellfinder.core import types
from magicgui import widgets
from magicgui.widgets import FunctionGui
from napari.layers import Image, Labels, Layer, Points
from napari.layers.utils.layer_utils import features_to_pandas_dataframe
from napari.qt.threading import WorkerBase

from cell_3d_vol_vis.main import CellsData, main


class Worker(WorkerBase):

    def __init__(self, **main_kwargs):
        super().__init__()
        self.main_kwargs = main_kwargs

    def work(
        self,
    ) -> tuple[tuple[int, int, int], list[list[Cell]], list[types.array]]:
        return main(**self.main_kwargs)


class VolumeVisWidget:

    def __init__(self):
        self.layers: list[Layer] = []
        self.viewer: napari.Viewer | None = None
        self.seg_layer: Image | None = None

        self.worker: Worker | None = None

        self.run_widget: FunctionGui | None = None
        self.layers_widget: FunctionGui | None = None
        self.segmentation_layer_widget: FunctionGui | None = None

        self._widgets_that_list_layers: list[widgets.ListEdit] = []

    def _process_image_layers(
        self,
        data: list[types.array],
        layers: list[Image],
        cuboid_offset: tuple[int, int, int],
        voxel_size: tuple[float, float, float],
        update_existing_layers: bool,
        scale_to_voxel: bool,
    ):
        for image, image_layer in zip(data, layers, strict=False):
            name = f"vol-{image_layer.name}"
            if update_existing_layers and name in self.viewer.layers:
                new_layer = self.viewer.layers[name]
                new_layer.data = image
                new_layer.translate = cuboid_offset
            else:
                new_layer = self.viewer.add_image(
                    data=image,
                    name=name,
                    translate=cuboid_offset,
                )
                new_layer.visible = image_layer.visible
                for prop in (
                    "opacity",
                    "colormap",
                    "contrast_limits",
                    "contrast_limits_range",
                    "gamma",
                    "attenuation",
                    "rendering",
                    "blending",
                ):
                    setattr(new_layer, prop, getattr(image_layer, prop))

            if scale_to_voxel:
                new_layer.translate = [
                    o * v
                    for o, v in zip(cuboid_offset, voxel_size, strict=False)
                ]
                new_layer.scale = voxel_size

    def _process_label_layers(
        self,
        data: list[types.array],
        layers: list[Labels],
        cuboid_offset: tuple[int, int, int],
        voxel_size: tuple[float, float, float],
        update_existing_layers: bool,
        scale_to_voxel: bool,
    ):
        for image, image_layer in zip(data, layers, strict=False):
            name = f"vol-{image_layer.name}"
            if update_existing_layers and name in self.viewer.layers:
                new_layer = self.viewer.layers[name]
                new_layer.data = image
                new_layer.translate = cuboid_offset
            else:
                new_layer = self.viewer.add_labels(
                    data=image,
                    name=name,
                    translate=cuboid_offset,
                )

                new_layer.visible = image_layer.visible
                for prop in (
                    "opacity",
                    "colormap",
                    "contrast_limits",
                    "contrast_limits_range",
                    "gamma",
                    "attenuation",
                    "rendering",
                    "blending",
                ):
                    if hasattr(new_layer, prop):
                        setattr(new_layer, prop, getattr(image_layer, prop))

            if scale_to_voxel:
                new_layer.translate = [
                    o * v
                    for o, v in zip(cuboid_offset, voxel_size, strict=False)
                ]
                new_layer.scale = voxel_size

    def _process_cell_layers(
        self,
        cells: list[CellsData],
        layers: list[Points],
        voxel_size: tuple[float, float, float],
        update_existing_layers: bool,
        scale_to_voxel: bool,
    ):
        array_props = {
            "symbol",
            "face_color",
            "size",
            "border_width",
            "border_color",
        }
        for layer_cells, cell_layer in zip(cells, layers, strict=False):
            name = f"vol-{cell_layer.name}"
            mask = layer_cells["mask"]

            if update_existing_layers and name in self.viewer.layers:
                new_layer = self.viewer.layers[name]
                new_layer.selected_data = []
                # there's weird graphics sometimes when re-using points layer
                new_layer.data = []
                new_layer.data = layer_cells["pos"][mask, :]
                new_layer.features = layer_cells["features"].loc[mask, :]
            else:
                new_layer = self.viewer.add_points(
                    data=layer_cells["pos"][mask, :],
                    features=layer_cells["features"].loc[mask, :],
                    name=name,
                    n_dimensional=True,
                    visible=True,
                )

                new_layer.visible = cell_layer.visible
                for prop in (
                    "opacity",
                    "symbol",
                    "current_symbol",
                    "face_color",
                    "current_face_color",
                    "size",
                    "current_size",
                    "text",
                    "border_width",
                    "current_border_width",
                    "border_color",
                    "current_border_color",
                    "border_colormap",
                    "border_contrast_limits",
                    "face_colormap",
                    "face_contrast_limits",
                    "out_of_slice_display",
                    "shading",
                    "antialiasing",
                    "canvas_size_limits",
                ):
                    prop_val = getattr(cell_layer, prop)
                    if prop in array_props and len(prop_val):
                        setattr(new_layer, prop, prop_val[mask])
                    else:
                        setattr(new_layer, prop, prop_val)

            if scale_to_voxel:
                new_layer.scale = voxel_size
            new_layer.refresh()

    def process_worker_result(
        self,
        result: tuple[
            tuple[int, int, int],
            list[CellsData],
            list[types.array],
        ],
        voxel_size: tuple[float, float, float],
        show_in_3d: bool,
        update_existing_layers: bool,
        scale_to_voxel: bool,
        center: tuple[int, int, int],
    ):
        cuboid_offset, cells, images = result

        label_or_image = [
            lr for lr in self.layers if not isinstance(lr, Points)
        ]
        img_indices = [
            i for i, lr in enumerate(label_or_image) if isinstance(lr, Image)
        ]
        self._process_image_layers(
            [images[i] for i in img_indices],
            [label_or_image[i] for i in img_indices],
            cuboid_offset,
            voxel_size,
            update_existing_layers,
            scale_to_voxel,
        )
        label_indices = [
            i for i, lr in enumerate(label_or_image) if isinstance(lr, Labels)
        ]
        self._process_label_layers(
            [images[i] for i in label_indices],
            [label_or_image[i] for i in label_indices],
            cuboid_offset,
            voxel_size,
            update_existing_layers,
            scale_to_voxel,
        )
        self._process_cell_layers(
            cells,
            [lr for lr in self.layers if isinstance(lr, Points)],
            voxel_size,
            update_existing_layers,
            scale_to_voxel,
        )

        for layer in self.layers:
            layer.visible = False
        if self.seg_layer is not None:
            self.seg_layer.visible = False

        self.viewer.dims.set_current_step(0, center[0])
        self.viewer.camera.center = tuple(
            c * vx for c, vx in zip(center, voxel_size, strict=False)
        )

        if show_in_3d:
            self.view_3d()

    def view_2d(self):
        self.layers_widget()
        widget = self.run_widget
        viewer = self.viewer

        voxel_size = widget.voxel_size.get_value()

        center = (
            int(round(viewer.dims.current_step[0] * voxel_size[0])),
            int(viewer.camera.center[1]),
            int(viewer.camera.center[2]),
        )
        zoom = viewer.camera.zoom

        viewer.dims.ndisplay = 2
        viewer.camera.center = center
        viewer.camera.zoom = zoom

    def view_3d(self):
        self.layers_widget()
        widget = self.run_widget
        viewer = self.viewer

        voxel_size = widget.voxel_size.get_value()

        center = (
            int(round(viewer.dims.current_step[0] * voxel_size[0])),
            int(viewer.camera.center[1]),
            int(viewer.camera.center[2]),
        )
        zoom = viewer.camera.zoom
        orientation = viewer.camera.orientation

        viewer.dims.ndisplay = 3
        viewer.camera.center = center
        viewer.camera.zoom = zoom
        viewer.camera.orientation = orientation
        viewer.camera.angles = 0, 0, 90

    def apply_current_prop_to_existing_points(self):
        self.layers_widget()

        for layer in self.viewer.layers.selection:
            if not isinstance(layer, Points):
                continue

            layer.face_color[:] = mcolors.to_rgba(layer.current_face_color)
            layer.border_color[:] = mcolors.to_rgba(layer.current_border_color)
            layer.size[:] = layer.current_size
            layer.symbol[:] = layer.current_symbol
            layer.border_width[:] = layer.current_border_width

            layer.shading = "spherical"

            layer.refresh()

    def set_layers_voxel_sizes(self):
        self.layers_widget()
        layers = list(self.viewer.layers.selection)

        vox_size = self.run_widget.voxel_size.get_value()
        step = list(self.viewer.dims.current_step)

        for layer in layers:
            layer.scale = vox_size
        # refresh at end so we don't do it after every layer change
        for layer in layers:
            layer.refresh()

        self.viewer.dims.current_step = step

    def layers_run(
        self,
        viewer: napari.Viewer,
        layers: list[Layer],
    ):
        """
        magicgui widget for setting the signal_image parameter.

        Parameters
        ----------
        layers : Layer
             Hmm
        """
        self.layers = layers
        self.viewer = viewer

    def segmentation_layer_run(
        self,
        seg_layer: Image | None,
    ):
        """
        magicgui widget for setting the cell layer.

        Parameters
        ----------
        seg_layer : Image
            The cell layer containing the detected cells to analyse.
        """
        self.seg_layer = seg_layer

    def run(
        self,
        selected_cells_only: bool = False,
        voxel_size: tuple[float, float, float] = (5, 1, 1),
        cuboid_size: tuple[float, float, float] = (100, 50, 50),
        region_cuboids_path: Path | None = None,
        selected_region_id: int = -1,
        update_existing_layers: bool = False,
        show_in_3d: bool = False,
        available_points: Points | None = None,
        current_point: int = 0,
        center_at: Literal[
            "View Center", "Random Point", "Selected Point", "Next Point"
        ] = "View Center",
    ) -> None:
        """
        Run analysis.

        Parameters
        ----------
        """

        # we must manually call so that the parameters of these functions are
        # initialized and updated. Because, if the images are open in napari
        # before we open cellfinder, then these functions may never be called
        self.layers_widget()
        self.segmentation_layer_widget()
        layers = self.layers
        viewer: napari.Viewer = self.viewer

        if center_at == "View Center":
            center = (
                int(viewer.dims.current_step[0]),
                int(viewer.camera.center[1] / voxel_size[1]),
                int(viewer.camera.center[2] / voxel_size[2]),
            )
        else:
            if available_points is None:
                raise ValueError(
                    "Center points layer must be provided if centering the "
                    "cuboid at a point"
                )
            n = len(available_points.data)
            if not n:
                raise ValueError("There must be some points in points layer")

            if set(available_points.translate) != {0}:
                raise ValueError("Layer translate must all be zero")

            if center_at == "Random Point":
                i = random.randrange(n)
            elif center_at == "Next Point":
                if not (0 <= current_point < n):
                    raise ValueError(
                        f"Current point out of range "
                        f"({current_point} / [0, {n - 1}])"
                    )
                i = current_point

                current_point += 1
                if current_point == n:
                    current_point = 0
                self.run_widget.current_point.value = current_point
            else:
                selection = list(available_points.selected_data)
                if len(selection) != 1:
                    raise ValueError(
                        "Exactly one point must be selected in points layer"
                    )
                i = selection[0]
            center = list(map(int, available_points.data[i]))

        cells = []
        for layer in layers:
            if set(layer.translate) != {0}:
                raise ValueError("Layer translate must all be zero")

            if not isinstance(layer, Points):
                continue

            data = layer.data
            features = features_to_pandas_dataframe(layer.features)

            if selected_cells_only:
                mask = np.zeros(len(data), dtype=np.bool)
                mask[np.asarray(list(layer.selected_data))] = True
            else:
                mask = np.ones(len(data), dtype=np.bool)
            cells.append({"pos": data, "features": features, "mask": mask})

        cuboid_size_px = [
            int(round(size / um_ppx))
            for size, um_ppx in zip(cuboid_size, voxel_size, strict=False)
        ]

        start = [0, 0, 0]
        for i, (c, size) in enumerate(
            zip(center, cuboid_size_px, strict=False)
        ):
            d_start = c - size // 2
            if d_start < 0:
                cuboid_size_px[i] = max(d_start + cuboid_size_px[i], 0)
                d_start = 0

            start[i] = d_start

        worker = Worker(
            images=[
                layer.data for layer in layers if not isinstance(layer, Points)
            ],
            cells=cells,
            cuboid_offset=start,
            cuboid_size=cuboid_size_px,
            region_segmentation=(
                None if self.seg_layer is None else self.seg_layer.data
            ),
            selected_region_id=(
                None if selected_region_id < 0 else selected_region_id
            ),
            region_cuboids_path=region_cuboids_path,
        )
        self.worker = worker

        # Make sure if the worker emits an error, it is propagated to this
        # thread
        worker.errored.connect(reraise)
        worker.returned.connect(
            partial(
                self.process_worker_result,
                voxel_size=voxel_size,
                show_in_3d=show_in_3d,
                scale_to_voxel=True,
                update_existing_layers=update_existing_layers,
                center=center,
            )
        )

        worker.start()

    def _update_layer_widgets_new_layers(self, *args):
        for widget in self._widgets_that_list_layers:
            widget.reset_choices()

    def _update_current_point_from_layer_change(self, *args):
        self.run_widget.current_point.value = 0

    def build(self) -> widgets.Container:
        self.run_widget = FunctionGui(
            self.run,
            call_button="Run extract",
            persist=True,
            param_options={
                "cuboid_size": {"options": {"min": 0, "max": 10000}},
                "selected_region_id": {"min": -1, "max": 100_000},
            },
        )
        self.run_widget.available_points.changed.connect(
            self._update_current_point_from_layer_change
        )
        self.layers_widget = FunctionGui(
            self.layers_run,
            call_button=False,
            persist=False,
            scrollable=False,
            labels=False,
            auto_call=True,
            param_options={"layers": {"layout": "vertical"}},
        )
        self.segmentation_layer_widget = FunctionGui(
            self.segmentation_layer_run,
            call_button=False,
            persist=False,
            scrollable=False,
            labels=False,
            auto_call=True,
        )

        run_widget = self.run_widget
        for widget, new_name, insertion in zip(
            (self.layers_widget, self.segmentation_layer_widget),
            ("Data layers", "Segmentation layer"),
            ("selected_cells_only", "region_cuboids_path"),
            strict=True,
        ):
            # make it look as if it's directly in the root container
            widget.margins = 0, 0, 0, 0
            # the parameters of these widgets are updated using `auto_call`
            # only. If False, magicgui passes these as args to root() when the
            # root's function runs. But that doesn't list them as args of its
            # function
            widget.gui_only = True
            run_widget.insert(run_widget.index(insertion), widget)
            getattr(run_widget, widget.name).label = new_name

        container = widgets.Container(
            widgets=[
                run_widget,
                FunctionGui(
                    self.view_2d, call_button="Show 2D view", auto_call=False
                ),
                FunctionGui(
                    self.view_3d, call_button="Show 3D view", auto_call=False
                ),
                FunctionGui(
                    self.set_layers_voxel_sizes,
                    call_button="Set sel. layers' voxel size",
                    auto_call=False,
                ),
                FunctionGui(
                    self.apply_current_prop_to_existing_points,
                    call_button="Set sel. P layers from current",
                    auto_call=False,
                ),
            ],
            layout="vertical",
            labels=False,
            scrollable=True,
        )

        self._widgets_that_list_layers = [
            self.layers_widget.layers,
            self.segmentation_layer_widget.seg_layer,
            self.run_widget.available_points,
        ]

        viewer = napari.current_viewer()
        viewer.layers.events.inserted.connect(
            self._update_layer_widgets_new_layers
        )
        viewer.layers.events.removed.connect(
            self._update_layer_widgets_new_layers
        )
        viewer.layers.events.renamed.connect(
            self._update_layer_widgets_new_layers
        )

        # needed for enabling scrolling
        return container.root_native_widget


def reraise(e: Exception) -> None:
    """Re-raises the exception."""
    raise Exception from e


def cell_3d_vol_widget():
    vol_vis = VolumeVisWidget()
    return vol_vis.build()
