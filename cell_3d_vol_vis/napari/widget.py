""" """

from functools import partial
from pathlib import Path

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
        for layer_cells, cell_layer in zip(cells, layers, strict=False):
            name = f"vol-{cell_layer.name}"
            mask = layer_cells["mask"]

            if update_existing_layers and name in self.viewer.layers:
                new_layer = self.viewer.layers[name]
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
                    "face_color",
                    "size",
                    "text",
                    "border_width",
                    "border_color",
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
                    if isinstance(prop_val, np.ndarray):
                        setattr(new_layer, prop, prop_val[mask])
                    else:
                        setattr(new_layer, prop, prop_val)

            if scale_to_voxel:
                new_layer.scale = voxel_size

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
    ):
        step = list(self.viewer.dims.current_step)
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

        self.viewer.dims.current_step = step

        if show_in_3d:
            self.view_3d()

    def view_2d(self):
        self.layers_widget()
        widget = self.run_widget
        viewer = self.viewer

        do_scale = widget.scale_to_voxel.get_value()
        voxel_size = widget.voxel_size.get_value()

        center = (
            int(viewer.dims.current_step[0]),
            int(viewer.camera.center[1]),
            int(viewer.camera.center[2]),
        )
        if do_scale:
            center = center[0] * voxel_size[0], *center[1:]
        zoom = viewer.camera.zoom

        viewer.dims.ndisplay = 2
        viewer.camera.center = center
        viewer.camera.zoom = zoom

    def view_3d(self):
        self.layers_widget()
        widget = self.run_widget
        viewer = self.viewer

        do_scale = widget.scale_to_voxel.get_value()
        voxel_size = widget.voxel_size.get_value()

        center = (
            int(viewer.dims.current_step[0]),
            int(viewer.camera.center[1]),
            int(viewer.camera.center[2]),
        )
        if do_scale:
            center = center[0] * voxel_size[0], *center[1:]
        zoom = viewer.camera.zoom
        orientation = viewer.camera.orientation

        viewer.dims.ndisplay = 3
        viewer.camera.center = center
        viewer.camera.zoom = zoom
        viewer.camera.orientation = orientation
        viewer.camera.angles = 0, 0, 90

    def set_layers_voxel_sizes(self):
        self.layers_widget()
        self.segmentation_layer_widget()

        vox_size = self.run_widget.voxel_size.get_value()
        step = list(self.viewer.dims.current_step)

        for layer in self.layers:
            layer.scale = vox_size
            layer.refresh()
        if self.seg_layer is not None:
            self.seg_layer.scale = vox_size
            self.seg_layer.refresh()

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
        scale_to_voxel: bool = False,
        update_existing_layers: bool = False,
        show_in_3d: bool = False,
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

        cells = []
        for layer in layers:
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

        center = (
            int(viewer.dims.current_step[0]),
            int(viewer.camera.center[1]),
            int(viewer.camera.center[2]),
        )
        if scale_to_voxel:
            center = (
                center[0],
                int(center[1] / voxel_size[1]),
                int(center[2] / voxel_size[2]),
            )
            cuboid_size_px = [
                int(round(size / um_ppx))
                for size, um_ppx in zip(cuboid_size, voxel_size, strict=False)
            ]
        else:
            cuboid_size_px = list(map(int, cuboid_size))

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
                scale_to_voxel=scale_to_voxel,
                update_existing_layers=update_existing_layers,
            )
        )

        worker.start()

    def build(self) -> widgets.Container:
        self.run_widget = FunctionGui(
            self.run,
            call_button=True,
            persist=True,
            param_options={
                "cuboid_size": {"options": {"min": 0, "max": 10000}},
                "selected_region_id": {"min": -1, "max": 100_000},
            },
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
                    self.set_layers_voxel_sizes,
                    call_button="Set layers voxel size",
                    auto_call=False,
                ),
                FunctionGui(
                    self.view_2d, call_button="Show in 2D", auto_call=False
                ),
                FunctionGui(
                    self.view_3d, call_button="Show in 3D", auto_call=False
                ),
            ],
            layout="vertical",
            labels=False,
        )

        return container


def reraise(e: Exception) -> None:
    """Re-raises the exception."""
    raise Exception from e


def cell_3d_vol_widget():
    vol_vis = VolumeVisWidget()
    return vol_vis.build()
