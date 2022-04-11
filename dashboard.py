import numpy as np
import pygfx as gfx
from pygfx.resources import Texture
from wgpu.gui.auto import WgpuCanvas
import pandas as pd
from mesmerize_napari.core import CaimanDataFrameExtensions, CaimanSeriesExtensions, \
    CNMFExtensions, MCorrExtensions, set_parent_data_path, get_parent_data_path
from skvideo.io import vread
from matplotlib import cm
from typing import *


CALCIUM_VIDEO_DIMS = (512, 512)
CALCIUM_VIDEO_COUNT = 4

BEHAVIOR_VIDEO_DIMS = (320, 228)
BEHAVIOR_VIDEO_COUNT = 2

SAMPLING_RATE_CALCIUM = 11.2
SAMPLING_RATE_BEHAVIOR = 500

FRAME_TIME_CALCIUM = 1_000.0 / SAMPLING_RATE_CALCIUM
FRAME_TIME_BEHAVIOR = 1_000.0 / SAMPLING_RATE_BEHAVIOR

TRIAL_DURATION = 10_000  # trial duration in milliseconds


class Dashboard:
    def __init__(self):
        # Create a canvas, scene, and rederer
        self.canvas = WgpuCanvas()
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()

        self.camera = gfx.OrthographicCamera(1800, 550)
        self.camera.position.y = 512
        self.camera.scale.y = -1
        self.camera.position.x = 1736 / 2

        gnuplot2_array = np.vstack([cm.gnuplot2(i) for i in range(256)])[:, 0:-1]
        gnuplot2_cmap = Texture(gnuplot2_array, dim=1).get_view()

        self.calcium_video_graphics: List[gfx.Image] = list()
        # create blank spots for the calcium vids
        for i in range(CALCIUM_VIDEO_COUNT):
            image_graphic = gfx.Image(
                gfx.Geometry(grid=self._get_blank_texture(CALCIUM_VIDEO_DIMS)),
                gfx.ImageBasicMaterial(clim=(3000, 6000), map=gnuplot2_cmap)
            )

            image_graphic.position.x = i * (50 + 512)

            self.calcium_video_graphics.append(image_graphic)
            self.scene.add(image_graphic)

        self.behavior_video_graphics: List[gfx.Image] = list()
        for i in range(BEHAVIOR_VIDEO_COUNT):
            image_graphic = gfx.Image(
                gfx.Geometry(grid=self._get_blank_texture(BEHAVIOR_VIDEO_DIMS)),
                gfx.ImageBasicMaterial(clim=(3000, 6000), map=gnuplot2_cmap)
            )

            image_graphic.position.x = i * (50 + 320)
            image_graphic.position.y = 512 + 16

            self.behavior_video_graphics.append(image_graphic)
            self.scene.add(image_graphic)

        self._trial_index: int = 0

        self._timepoint: float = 0.0

        self._calcium_memmaps: List[np.ndarray] = None
        self._behavior_arrays: List[np.ndarray] = None

        self._calcium_index = -1
        self._behavior_index = -1

        self._previous_calcium_index = -1
        self._previous_behavior_index = -1

        self._force_update: bool = False

    def _get_blank_texture(self, dims: Tuple[int, int]) -> gfx.Texture:
        return gfx.Texture(np.zeros(shape=dims, dtype=np.float32), dim=2)

    def animate(self):
        self.scene.traverse(self._update_scene())
        self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw()

    @property
    def timepoint(self) -> float:
        """
        timepoint in milliseconds
        Returns
        -------

        """
        return self._timepoint

    @timepoint.setter
    def timepoint(self, t: float):
        """
        Set the timepoint
        Parameters
        ----------
        t

        Returns
        -------

        """
        self._timepoint = t
        self._calcium_index = round(self._timepoint / FRAME_TIME_CALCIUM)
        self._behavior_index = round(self._timepoint / FRAME_TIME_BEHAVIOR)

    def set_session(self, animal_id: str, date: str, cell_type: str):
        # TODO: Set new calcium memmaps and behavior vids
        # TODO: Close the opened memmaps!!!

        self._calcium_index = 0
        self._behavior_index = 0

        self._previous_calcium_index = 0
        self._previous_behavior_index = 0

        self._force_update = True
        self._force_update = False

    def set_trial_index(self, index: int):
        self._trial_index = index

        self._calcium_index = int(round(self._timepoint / FRAME_TIME_CALCIUM) + (112 * self._trial_index))

    def _update_scene(self):
        self._update_calcium_frame()
        self._update_behavior_frame()

    def _update_calcium_frame(self):
        if self._calcium_memmaps is None:
            return

        if (self._calcium_index == self._previous_calcium_index) and not self._force_update:
            return

        for img_ix, img in enumerate(self.calcium_video_graphics):
            img.geometry.grid = gfx.Texture(
                self._calcium_memmaps[img_ix][self._calcium_index].T, dim=2
            )

    def _update_behavior_frame(self):
        if self._behavior_arrays is None:
            return

        if (self._behavior_index == self._previous_behavior_index) and not self._force_update:
            return

        for img_ix, img in enumerate(self.behavior_video_graphics):
            img.geometry.grid = gfx.Texture(
                self._behavior_arrays[img_ix][self._behavior_index], dim=2
            )
