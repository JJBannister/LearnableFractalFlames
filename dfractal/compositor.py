from typing import List

import taichi as ti
import taichi.math as tm

from .painter import Painter
from .utils import logistic


@ti.data_oriented
class Compositor:

    def __init__(self, painters: List[Painter], bg_color: tm.vec3 = None):

        self.painters = painters
        self.n_painters = len(painters)

        self.learn_bg_color = bg_color is None
        self.bg_color = bg_color
        self.bg_color_params = ti.Vector.field(
            n=3, dtype=float, shape=(), needs_grad=True
        )
        self.bg_color_params.fill(0.0)

        self.train_output_buffer = ti.Vector.field(
            n=3, dtype=float, shape=painters[0].train_rgba_buffer.shape, needs_grad=True
        )
        self.eval_output_buffer = ti.Vector.field(
            n=3, dtype=float, shape=painters[0].eval_rgba_buffer.shape, needs_grad=False
        )

    def forward_eval(self):
        self.composite_eval()

    def forward(self):
        self.clear()
        self.composite()

    def clear(self):
        self.train_output_buffer.fill(0)
        self.train_output_buffer.grad.fill(0)
        self.bg_color_params.grad.fill(0)

    def backward(self):
        self.composite.grad()

    @ti.kernel
    def update(self, lr: float):
        self.bg_color_params[None] -= lr * self.bg_color_params.grad[None]
        self.bg_color_params.grad.fill(0)

    @ti.kernel
    def composite(self):
        for x, y in self.train_output_buffer:
            color = tm.vec3(0.0)
            if self.learn_bg_color:
                color = logistic(self.bg_color_params[None])
            else:
                color = self.bg_color

            for l in ti.static(range(self.n_painters)):
                layer_rgba = self.painters[l].train_rgba_buffer[x, y]
                color = color * tm.max(1.0 - layer_rgba.a, 0.0) + layer_rgba.rgb

            self.train_output_buffer[x, y] = color

    @ti.kernel
    def composite_eval(self):
        for x, y in self.eval_output_buffer:
            color = tm.vec3(0.0)
            if self.learn_bg_color:
                color = logistic(self.bg_color_params[None])
            else:
                color = self.bg_color

            for l in ti.static(range(self.n_painters)):
                layer_rgba = self.painters[l].eval_rgba_buffer[x, y]
                color = color * tm.max(1.0 - layer_rgba.a, 0.0) + layer_rgba.rgb

            self.eval_output_buffer[x, y] = color
