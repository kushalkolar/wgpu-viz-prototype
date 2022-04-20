import numpy as np
from itertools import product
import pygfx
from typing import *


camera_types = {
    'o': pygfx.OrthographicCamera,
    'p': pygfx.PerspectiveCamera
}


class Subplot:
    def __init__(self):
        self.scene: pygfx.Scene = None
        self.camera: Union[pygfx.OrthographicCamera, pygfx.PerspectiveCamera] = None
        self.controller: pygfx.PanZoomController = None

    def add_graphic(self, graphic):
        self.scene.add(graphic)

        if isinstance(graphic, Image):
            dims = graphic.data.shape
            self.camera.set_view_size(*dims)
            self.camera.position.set(dims[0] / 2, dims[1] / 2, 0)


class GridPlot:
    def __init__(
            self, renderer: pygfx.Renderer,
            grid_shape: Tuple[int, int],
            cameras: np.ndarray,
            controllers: np.ndarray
    ):
        """

        Parameters
        ----------
        grid_shape:
            nrows, ncols

        controllers:
            numpy array of same shape as ``grid_shape`` that defines the controllers
            Example:
            unique controllers for a 2x2 gridplot: np.array([[0, 1], [2, 3]])
            same controllers for first 2 plots: np.array([[0, 0, 1], [2, 3, 4]])
        """

        if controllers.shape != grid_shape:
            raise ValueError

        if not np.all(np.sort(np.unique(controllers)) == np.arange(np.unique(controllers).size)):
            raise ValueError("controllers must be consecutive integers")

        nrows, ncols = grid_shape

        self.subplots: np.ndarray[Subplot] = np.ndarray(shape=(nrows, ncols), dtype=object)

        self.viewports: np.ndarray[Subplot] = np.ndarray(shape=(nrows, ncols), dtype=object)

        self._controllers: List[pygfx.PanZoomController] = [
            pygfx.PanZoomController() for i in range(np.unique(controllers).size)
        ]

        for i, j in product(range(nrows), range(ncols)):
            self.subplots[i, j].scene = pygfx.Scene()
            self.subplots[i, j].controller = self._controllers[controllers[i, j]]
            self.subplots[i, j].camera = camera_types.get(cameras[i, j])()
            self.viewports[i, j] = pygfx.Viewport(renderer)
            self.subplots[i, j].controller.add_default_event_handlers(
                self.viewports[i, j],
                self.subplots[i, j].camera
            )


class Image:
    def __init__(self, data: np.ndarray, vmin: int, vmax: int, cmap: str = 'plasma'):
        self.data = data
        self._image = pygfx.Image(
            pygfx.Geometry(grid=pygfx.Texture(data, dim=2)),
            pygfx.ImageBasicMaterial(clim=(vmin, vmax), map=getattr(pygfx.cm, cmap))
        )

    def update_data(self, data: np.ndarray):
        self._image.geometry.data[:] = data
        self._image.geometry.update_range((0, 0, 0), self._image.geometry.grid.size)
