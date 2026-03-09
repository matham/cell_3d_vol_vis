import logging
import pandas as pd
import numpy as np
from typing import TypedDict
from datetime import datetime
from brainglobe_utils.cells.cells import Cell
from cellfinder.core import types
from pathlib import Path
import yaml


class CellsData(TypedDict):
    pos: np.ndarray
    features: pd.DataFrame
    mask: np.ndarray


def extract_cuboid(
    images: list[types.array],
    cells: list[list[Cell] | CellsData],
    cuboid_offset: tuple[int, int, int],
    cuboid_size: tuple[int, int, int],
) -> tuple[
    tuple[int, int, int],
    list[list[Cell] | CellsData],
    list[types.array],
]:
    z_s, y_s, x_s = cuboid_offset
    z_e, y_e, x_e = [s + n for s, n in zip(cuboid_offset, cuboid_size)]

    new_cells = []
    for input_cells in cells:
        if isinstance(input_cells, (list, tuple)):
            new_cells.append(
                [c for c in input_cells if x_s <= c.x < x_e and y_s <= c.y < y_e and z_s <= c.z < z_e]
            )
        else:
            # order is z, y, x
            pos = input_cells["pos"]
            mask = input_cells["mask"]
            mask = (
                mask &
                (z_s <= pos[:, 0]) & (pos[:, 0] < z_e) &
                (y_s <= pos[:, 1]) & (pos[:, 1] < y_e) &
                (x_s <= pos[:, 2]) & (pos[:, 2] < x_e)
            )
            new_cells.append({"pos": pos, "features": input_cells["features"], "mask": mask})

    new_images = [np.array(im[z_s: z_e, y_s: y_e, x_s: x_e]) for im in images]
    return cuboid_offset, new_cells, new_images


def extract_region(
    images: list[types.array],
    cells: list[list[Cell] | CellsData],
    region_segmentation: types.array,
    region_cuboids_path: Path,
    selected_region_id: int,
) -> tuple[tuple[int, int, int], list[list[Cell] | CellsData], list[types.array]]:
    with open(region_cuboids_path, "r") as fh:
        region_boundaries = yaml.load(fh, yaml.Loader)
    (z_s, z_e), (y_s, y_e), (x_s, x_e) = region_boundaries[selected_region_id]

    new_cells = []
    for input_cells in cells:
        if isinstance(input_cells, (list, tuple)):
            new_cells.append(
                [c for c in input_cells if c.metadata["region_id"] == selected_region_id]
            )
        else:
            features = input_cells["features"]
            mask = input_cells["mask"]
            mask = mask & np.asarray(features["region_id"] == selected_region_id)
            new_cells.append({"pos": input_cells["pos"], "features": features, "mask": mask})

    seg = np.asarray(region_segmentation[z_s: z_e, y_s: y_e, x_s: x_e])
    seg_mask = seg != selected_region_id

    new_images = []
    for im in images:
        im = np.array(im[z_s: z_e, y_s: y_e, x_s: x_e], dtype=im.dtype, copy=True)
        im[seg_mask] = 0
        new_images.append(im)

    return (z_s, y_s, x_s), new_cells, new_images


def main(
    *,
    images: list[types.array],
    cells: list[list[Cell] | CellsData],
    cuboid_offset: tuple[int, int, int],
    cuboid_size: tuple[int, int, int],
    region_segmentation: types.array | None = None,
    selected_region_id: int | None = None,
    region_cuboids_path: Path | None = None
) -> tuple[tuple[int, int, int], list[list[Cell] | CellsData], list[types.array]]:
    """
    We expect the input data to have dimension order of z, y, x. All the
    parameters are specified in this order.
    """
    ts = datetime.now()

    if region_segmentation is not None and selected_region_id is not None and region_cuboids_path is not None:
        res = extract_region(images, cells, region_segmentation, region_cuboids_path, selected_region_id)
    else:
        res = extract_cuboid(images, cells, cuboid_offset, cuboid_size)

    logging.info(f"cell_3d_vol_vis: Volume extraction took {datetime.now() - ts}")

    return res


def run_main():
    raise NotImplementedError


if __name__ == "__main__":
    run_main()
