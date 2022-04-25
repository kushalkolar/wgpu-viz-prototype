import numpy as np
from itertools import product
import pygfx
from typing import *
from functools import partial

camera_types = {
    'o': pygfx.OrthographicCamera,
    'p': pygfx.PerspectiveCamera
}


class Subplot:
    def __init__(self):
        self.scene: pygfx.Scene = None
        self.camera: Union[pygfx.OrthographicCamera, pygfx.PerspectiveCamera] = None
        self.controller: pygfx.PanZoomController = None
        self.viewport: pygfx.Viewport  # might be better as an attribute of GridPlot
                                       # but easier to iterate when in same object as camera and scene
        self.position: Tuple[int, int] = None
        self.get_rect: callable = None

    def add_graphic(self, graphic):
        self.scene.add(graphic.world_object)

        if isinstance(graphic, Image):
            dims = graphic.data.shape
            self.camera.set_view_size(*dims)
            self.camera.position.set(dims[0] / 2, dims[1] / 2, 0)
            self.controller.update_camera(self.camera)


def _produce_rect(j, i, ncols, nrows, w, h):
    return [
        ((w / ncols) + ((i - 1) * (w / ncols))),
        ((h / nrows) + ((j - 1) * (h / nrows))),
        (w / ncols),
        (h / nrows)
    ]


class GridPlot:
    def __init__(
            self,
            canvas,
            renderer: pygfx.Renderer,
            grid_shape: Tuple[int, int],
            cameras: np.ndarray,
            controllers: np.ndarray
    ):
        """

        Parameters
        ----------
        grid_shape:
            nrows, ncols

        cameras: np.ndarray
            Array of ``o`` and/or ``p`` that specifies camera type for each subplot:
            ``o``: ``pygfx.OrthographicCamera``
            ``p``: ``pygfx.PerspectiveCamera``

        controllers:
            numpy array of same shape as ``grid_shape`` that defines the controllers
            Example:
            unique controllers for a 2x2 gridplot: np.array([[0, 1], [2, 3]])
            same controllers for first 2 plots: np.array([[0, 0, 1], [2, 3, 4]])
        """

        if controllers.shape != grid_shape:
            raise ValueError

        if cameras.shape != grid_shape:
            raise ValueError

        if not np.all(np.sort(np.unique(controllers)) == np.arange(np.unique(controllers).size)):
            raise ValueError("controllers must be consecutive integers")

        self.canvas = canvas
        self.renderer = renderer

        self.grid_shape = grid_shape
        nrows, ncols = self.grid_shape

        self.subplots: np.ndarray[Subplot] = np.ndarray(shape=(nrows, ncols), dtype=object)
        # self.viewports: np.ndarray[Subplot] = np.ndarray(shape=(nrows, ncols), dtype=object)

        self._controllers: List[pygfx.PanZoomController] = [
            pygfx.PanZoomController() for i in range(np.unique(controllers).size)
        ]

        for i, j in self._get_iterator():
            self.subplots[i, j] = Subplot()
            self.subplots[i, j].position = (i, j)

            self.subplots[i, j].scene = pygfx.Scene()
            self.subplots[i, j].controller = self._controllers[controllers[i, j]]
            self.subplots[i, j].camera = camera_types.get(cameras[i, j])()

            self.subplots[i, j].viewport = pygfx.Viewport(renderer)

            # self.viewports[i, j] = pygfx.Viewport(renderer)

            self.subplots[i, j].controller.add_default_event_handlers(
                self.subplots[i, j].viewport,
                self.subplots[i, j].camera
            )

            self.subplots[i, j].get_rect = partial(_produce_rect, i, j, ncols, nrows)

        self._animate_funcs: List[callable] = list()
        self._current_iter = None

    def animate(self):
        for subplot in self:
            subplot.controller.update_camera(subplot.camera)

        w, h = self.canvas.get_logical_size()

        for subplot in self:
            subplot.viewport.rect = subplot.get_rect(w, h)
            subplot.viewport.render(subplot.scene, subplot.camera)

        for f in self._animate_funcs:
            f()

        self.renderer.flush()
        self.canvas.request_draw()

    def _get_iterator(self):
        return product(range(self.grid_shape[0]), range(self.grid_shape[1]))

    def __iter__(self):
        self._current_iter = self._get_iterator()
        return self

    def __next__(self) -> Subplot:
        pos = self._current_iter.__next__()
        return self.subplots[pos]


class Image:
    def __init__(self, data: np.ndarray, vmin: int, vmax: int, cmap: str = 'plasma'):
        self.data = data
        self.world_object = pygfx.Image(
            pygfx.Geometry(grid=pygfx.Texture(data, dim=2)),
            pygfx.ImageBasicMaterial(clim=(vmin, vmax), map=getattr(pygfx.cm, cmap))
        )

    def update_data(self, data: np.ndarray):
        self.world_object.geometry.data[:] = data
        self.world_object.geometry.update_range((0, 0, 0), self.world_object.geometry.grid.size)
