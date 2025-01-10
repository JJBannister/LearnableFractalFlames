from typing import List
from abc import abstractmethod

import taichi as ti
import taichi.math as tm

from .sampler import Sampler


@ti.data_oriented
class FlameSampler(Sampler):

    def __init__(
        self,
        n_generators: int,
        train_n_threads: int,
        train_n_iters: int,
        eval_n_threads: int,
        eval_n_iters: int,
        generator_weights: List[float] = None,
    ):

        super().__init__(
            n_generators=n_generators,
            train_n_threads=train_n_threads,
            train_n_iters=train_n_iters,
            eval_n_threads=eval_n_threads,
            eval_n_iters=eval_n_iters,
            generator_weights=generator_weights,
        )

        # Parameter fields for affine transformations
        self.matrices = ti.Matrix.field(
            n=2, m=2, dtype=float, shape=(n_generators), needs_grad=True
        )
        self.translations = ti.Vector.field(
            n=2, dtype=float, shape=(n_generators), needs_grad=True
        )
        self.initialize_parameters_randomly()

    def clear(self):
        super().clear()
        self.matrices.grad.fill(0)
        self.translations.grad.fill(0)

    @ti.kernel
    def update(self, lr: float):
        for i in self.matrices:
            self.matrices[i] -= lr * self.matrices.grad[i]
            self.translations[i] -= lr * self.translations.grad[i]

    @ti.func
    def apply_linear_transform(
        self, generator_index: int, old_sample: tm.vec2
    ) -> tm.vec2:
        return (
            self.matrices[generator_index] @ old_sample
            + self.translations[generator_index]
        )

    @ti.kernel
    def initialize_parameters_randomly(self):
        for g in range(self.n_generators):
            self.translations[g] = tm.vec2(
                [
                    2.5 * (ti.random() - 0.5),
                    2.5 * (ti.random() - 0.5),
                ]
            )

            v = 1.5 * (ti.random() - 0.5)
            self.matrices[g] = tm.mat2(
                [
                    [v, 0],
                    [0, v],
                ]
            )

    @ti.func
    @abstractmethod
    def compute_sample_position(self, generator_index: int, old_sample: tm.vec2):
        pass


class LinearFlameSampler(FlameSampler):

    def __init__(
        self,
        n_generators: int,
        train_n_threads: int,
        train_n_iters: int,
        eval_n_threads: int,
        eval_n_iters: int,
        generator_weights: List[float] = None,
    ):

        super().__init__(
            n_generators=n_generators,
            train_n_threads=train_n_threads,
            train_n_iters=train_n_iters,
            eval_n_threads=eval_n_threads,
            eval_n_iters=eval_n_iters,
            generator_weights=generator_weights,
        )

    @ti.func
    def compute_sample_position(self, generator_index: int, old_sample: tm.vec2):
        return self.apply_linear_transform(generator_index, old_sample)
