from typing import Tuple

import taichi as ti
import taichi.math as tm

from .splatter import Splatter
from .palette import Palette


@ti.data_oriented
class Painter:

    def __init__(self, splatter: Splatter, palette: Palette):

        self.splatter = splatter
        self.palette = palette

        # Output buffers
        self.train_rgba_buffer = ti.Vector.field(
            n=4, dtype=float, shape=splatter.train_splat_buffer.shape, needs_grad=True
        )

        self.eval_rgba_buffer = ti.Vector.field(
            n=4, dtype=float, shape=splatter.eval_splat_buffer.shape, needs_grad=False
        )

    def forward_eval(self):
        self.paint_eval()

    def forward(self):
        self.clear()
        self.paint()

    def clear(self):
        self.palette.clear()
        self.train_rgba_buffer.fill(0)
        self.train_rgba_buffer.grad.fill(0)

    def backward(self):
        self.paint.grad()

    @ti.kernel
    def paint(self):
        for x, y in self.splatter.train_splat_buffer:
            max_weight = self.splatter.train_max_weight[None]
            pixel_quality = self.splatter.train_splat_buffer[x, y]

            pixel_rgba = self.palette.compute_rgba(pixel_quality, max_weight)
            self.train_rgba_buffer[x, y] = pixel_rgba

    @ti.kernel
    def paint_eval(self):
        for x, y in self.splatter.eval_splat_buffer:
            max_weight = self.splatter.eval_max_weight[None]
            pixel_quality = self.splatter.eval_splat_buffer[x, y]

            pixel_rgba = self.palette.compute_rgba(pixel_quality, max_weight)
            self.eval_rgba_buffer[x, y] = pixel_rgba
