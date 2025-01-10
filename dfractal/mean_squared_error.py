import numpy as np
import taichi as ti
import taichi.math as tm


@ti.data_oriented
class MeanSquaredError:
    def __init__(self, image_buffer: ti.Vector.field):

        self.image_buffer = image_buffer

        image_shape = image_buffer.shape
        self.n_pixels = float(image_shape[0] * image_shape[1])

        self.reference_buffer = ti.Vector.field(n=3, dtype=float, shape=image_shape)
        self.loss = ti.field(float, shape=(), needs_grad=True)

    def forward(self):
        self.clear()
        self.compute_loss()

    def clear(self):
        self.loss.fill(0)

    def backward(self):
        self.loss.grad.fill(1.0)
        self.compute_loss.grad()

    def set_reference_image(self, reference_image: np.array):
        self.reference_buffer.from_numpy(reference_image)

    @ti.kernel
    def compute_loss(self):
        for i, j in self.image_buffer:
            true_pixel = self.image_buffer[i, j]
            reference_pixel = self.reference_buffer[i, j]
            d = ((true_pixel - reference_pixel) ** 2) / self.n_pixels
            self.loss[None] += d.sum()
