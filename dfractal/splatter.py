from typing import Tuple

from random import random

import taichi as ti
import taichi.math as tm

from .sampler import Sampler


@ti.data_oriented
class Splatter:

    def __init__(
        self,
        sampler: Sampler,
        train_resolution: Tuple[int] = (128, 128),
        eval_resolution: Tuple[int] = (1024, 1024),
    ):

        self.sampler = sampler
        self.train_resolution = train_resolution
        self.eval_resolution = eval_resolution

        # Dtype for sample quality
        self.quality_vec = ti.types.vector(n=self.sampler.n_generators, dtype=float)

        # Parameter to control the magnitude of sample quality blending
        self.quality_blend_param = ti.field(float, shape=(), needs_grad=True)

        # Parameters for final transform
        self.matrix = ti.Matrix.field(n=2, m=2, dtype=float, shape=(), needs_grad=True)
        self.translation = ti.Vector.field(n=2, dtype=float, shape=(), needs_grad=True)

        # Initialize the final transform
        self.matrix[None] = tm.mat2(
            [
                [1, 0],
                [0, 1],
            ]
        )
        self.translation[None] = tm.vec2([random() - 0.5, random() - 0.5])

        # Output buffers
        self.train_splat_buffer = ti.Vector.field(
            n=sampler.n_generators, dtype=float, shape=train_resolution, needs_grad=True
        )
        self.eval_splat_buffer = ti.Vector.field(
            n=sampler.n_generators, dtype=float, shape=eval_resolution, needs_grad=False
        )

        # Maximum pixel weight
        self.train_max_weight = ti.field(dtype=float, shape=())
        self.eval_max_weight = ti.field(dtype=float, shape=())

    def forward_eval(self):
        self.splat_eval()
        self.find_max_weight_eval()

    def forward(self):
        self.clear()
        self.splat()
        self.find_max_weight()

    def clear(self):
        self.train_max_weight.fill(0)

        self.train_splat_buffer.fill(0)
        self.train_splat_buffer.grad.fill(0)

        self.matrix.grad.fill(0)
        self.translation.grad.fill(0)
        self.quality_blend_param.grad.fill(0)

    def backward(self):
        self.splat.grad()

    @ti.kernel
    def update(self, lr: float):
        self.matrix[None] -= lr * self.matrix.grad[None]
        self.translation[None] -= lr * self.translation.grad[None]
        self.quality_blend_param[None] -= 5.0 * lr * self.quality_blend_param.grad[None]

    @ti.kernel
    def find_max_weight(self):
        for x, y in self.train_splat_buffer:
            weight = self.train_splat_buffer[x, y].sum()
            ti.atomic_max(self.train_max_weight[None], weight)

    @ti.kernel
    def find_max_weight_eval(self):
        for x, y in self.eval_splat_buffer:
            weight = self.eval_splat_buffer[x, y].sum()
            ti.atomic_max(self.eval_max_weight[None], weight)

    @ti.kernel
    def splat(self):
        for i, j in self.sampler.sample_positions:
            if j > 20:

                # Get the quality of the sample
                quality_blend_factor = 1.0 + tm.exp(self.quality_blend_param[None])
                quality = self.quality_vec(0.0)

                # Only look back at 10 most recent samples
                for k in ti.static(range(10)):
                    g = self.sampler.generator_indices[i, j - k]
                    quality_sample = self.quality_vec(0.0)
                    quality_sample[g] = 1.0
                    quality += quality_sample / quality_blend_factor**k

                # Map sample to screen and splat
                position = self.sampler.sample_positions[i, j]
                screen_position = self.to_screen(
                    self.final_transform(position), train=True
                )
                center_pixel = int(screen_position + 0.5)

                for x in ti.static(range(3)):
                    for y in ti.static(range(3)):
                        pixel = int(
                            tm.vec2(
                                [
                                    center_pixel.x + x - 1,
                                    center_pixel.y + y - 1,
                                ]
                            )
                        )

                        if self.is_pixel_valid(pixel, train=True):
                            weight = self.splat_weight(screen_position, pixel)
                            self.train_splat_buffer[pixel.x, pixel.y] += (
                                weight * quality
                            )

    @ti.kernel
    def splat_eval(self):
        for i in range(self.sampler.eval_n_threads):
            position = tm.vec2(0.0)
            quality = self.quality_vec(0.0)

            for j in range(self.sampler.eval_n_iters):
                g = self.sampler.compute_generator_index()
                position = self.sampler.compute_sample_position(g, position)

                # Compute new sample quality
                quality_blend_factor = 1.0 + tm.exp(self.quality_blend_param[None])
                quality_sample = self.quality_vec(0.0)
                quality_sample[g] = 1.0
                quality = quality / quality_blend_factor + quality_sample

                # accumulate after 20 iters if pixel is valid
                if i > 20:
                    screen_position = self.to_screen(
                        self.final_transform(position), train=False
                    )
                    pixel = int(screen_position + 0.5)

                    if self.is_pixel_valid(pixel, train=False):
                        self.eval_splat_buffer[pixel.x, pixel.y] += quality

    @ti.func
    def final_transform(self, p: tm.vec2) -> tm.vec2:
        return self.matrix[None] @ p + self.translation[None]

    @ti.func
    def to_screen(self, p: tm.vec2, train: bool) -> tm.vec2:
        """Maps [-1,1]^2 to pixel coordinates"""
        screen_position = tm.vec2(0.0)

        if train:
            screen_position = tm.vec2(
                [
                    (p.x + 1.0) * self.train_resolution[0] / 2.0,
                    (p.y + 1.0) * self.train_resolution[1] / 2.0,
                ]
            )

        else:
            screen_position = tm.vec2(
                [
                    (p.x + 1.0) * self.eval_resolution[0] / 2.0,
                    (p.y + 1.0) * self.eval_resolution[1] / 2.0,
                ]
            )

        return screen_position

    @ti.func
    def is_pixel_valid(self, pix: tm.vec2, train: bool):
        is_valid = False
        if train:
            is_valid = (
                (pix.x >= 0)
                and (pix.x < self.train_resolution[0])
                and (pix.y >= 0)
                and (pix.y < self.train_resolution[1])
            )
        else:
            is_valid = (
                (pix.x >= 0)
                and (pix.x < self.eval_resolution[0])
                and (pix.y >= 0)
                and (pix.y < self.eval_resolution[1])
            )
        return is_valid

    @ti.func
    def splat_weight(self, p: tm.vec2, pixel: tm.vec2) -> float:
        d_squared = ((p - float(pixel)) ** 2).sum()
        return tm.exp(-d_squared / (0.5))
