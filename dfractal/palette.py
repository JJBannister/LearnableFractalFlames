from typing import Tuple

import taichi as ti
import taichi.math as tm

from .utils import logistic


@ti.data_oriented
class Palette:

    def __init__(self, n_qualities: int, greyscale_mode: bool = False):
        self.greyscale_mode = greyscale_mode
        self.n_qualities = n_qualities

        self.quality_vec = ti.types.vector(n=self.n_qualities, dtype=float)
        self.quality_rgba = ti.Vector.field(
            n=4, dtype=float, shape=(self.n_qualities), needs_grad=True
        )

        self.initialize_parameters_randomly()

    def clear(self):
        self.quality_rgba.grad.fill(0.0)

    @ti.kernel
    def update(self, lr: float):
        for i in range(self.n_qualities):
            self.quality_rgba[i] -= lr * self.quality_rgba.grad[i]
        self.quality_rgba.grad.fill(0)

    @ti.func
    def compute_rgba(self, quality_vec, max_weight) -> tm.vec4:
        weight = 1 + quality_vec.sum()
        log_weight = tm.log(weight)
        log_max_weight = tm.log(1 + max_weight)

        weight_alpha = log_weight / log_max_weight

        rgba = tm.vec4(0)
        if self.greyscale_mode:
            rgba = tm.vec4(1.0) * weight_alpha

        else:
            for i in ti.static(range(self.n_qualities)):
                rgba += (
                    logistic(self.quality_rgba[i])
                    * quality_vec[i]
                    * weight_alpha
                    / weight
                )

        # Pre-multiply alpha channel
        rgba = tm.vec4(rgba.rgb * rgba.a, rgba.a)
        return rgba

    @ti.kernel
    def initialize_parameters_randomly(self):
        for i in range(self.n_qualities):
            self.quality_rgba[i] = tm.vec4(
                [
                    ti.random(),
                    ti.random(),
                    ti.random(),
                    0,
                ]
            )
