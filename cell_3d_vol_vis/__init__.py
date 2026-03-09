try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from .napari.widget import (
    cell_3d_vol_widget,
)

__all__ = ("cell_3d_vol_widget",)
