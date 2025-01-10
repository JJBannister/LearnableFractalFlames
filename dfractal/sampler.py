from typing import List
from itertools import accumulate
from abc import ABC, abstractmethod

import taichi as ti
import taichi.math as tm


@ti.data_oriented
class Sampler(ABC):

    def __init__(
        self,
        n_generators: int,
        train_n_threads: int,
        train_n_iters: int,
        eval_n_threads: int,
        eval_n_iters: int,
        generator_weights: List[float] = None,
    ):

        self.train_n_threads = train_n_threads
        self.train_n_iters = train_n_iters
        self.eval_n_threads = eval_n_threads
        self.eval_n_iters = eval_n_iters

        # Use equal weighting if weights are not provided
        if generator_weights is None:
            generator_weights = [1.0] * n_generators

        # Perform random sampling to select the generator function for each thread/iteration
        assert n_generators == len(
            generator_weights
        ), "generator weights does not match number of generators"
        self.n_generators = n_generators
        weight_sum = sum(generator_weights)
        self.generator_thresholds = [
            x / weight_sum for x in accumulate(generator_weights)
        ]

        self.generator_indices = ti.field(
            ti.int8, shape=(self.train_n_threads, self.train_n_iters)
        )
        self.compute_generator_indices()

        # Define data container for sample positions
        self.sample_positions = ti.Vector.field(
            n=2,
            dtype=float,
            shape=(self.train_n_threads, self.train_n_iters),
            needs_grad=True,
        )

    def forward(self):
        self.clear()
        self.compute_sample_positions()

    def clear(self):
        self.sample_positions.fill(0.0)
        self.sample_positions.grad.fill(0.0)

    def backward(self):
        self.compute_sample_positions.grad()

    @ti.kernel
    @abstractmethod
    def update(self, lr: float):
        pass

    @ti.func
    @abstractmethod
    def compute_sample_position(self, generator_index: int, old_sample: tm.vec2):
        pass

    @ti.kernel
    def compute_generator_indices(self):
        for t, i in ti.ndrange(self.train_n_threads, self.train_n_iters):
            self.generator_indices[t, i] = self.compute_generator_index()

    @ti.kernel
    def compute_sample_positions(self):
        for t in range(self.train_n_threads):
            for i in range(1, self.train_n_iters):
                old_sample = self.sample_positions[t, i - 1]
                g = self.generator_indices[t, i]
                self.sample_positions[t, i] = self.compute_sample_position(
                    g, old_sample
                )

    @ti.func
    def compute_generator_index(self):
        r = ti.random()
        g = 0
        for i in ti.static(range(self.n_generators)):
            thresh = self.generator_thresholds[i]
            if r < thresh:
                g = i
                r = float("inf")
        return g
