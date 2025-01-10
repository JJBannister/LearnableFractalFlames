from typing import List, Tuple
from abc import ABC, abstractmethod
import taichi as ti

from .sampler import Sampler
from .splatter import Splatter
from .painter import Painter
from .palette import Palette
from .compositor import Compositor
from .mean_squared_error import MeanSquaredError


class Pipeline(ABC):
    resolution: Tuple[int]
    output_buffer: ti.Vector.field

    samplers: List[Sampler]
    splatters: List[Splatter]
    palettes: List[Palette]
    painters: List[Painter]
    compositor: Compositor
    mse: MeanSquaredError

    def forward_eval(self):
        for i in range(len(self.samplers)):
            self.splatters[i].forward_eval()
            self.painters[i].forward_eval()
        self.compositor.forward_eval()

    def forward(self):
        for i in range(len(self.samplers)):
            self.samplers[i].forward()
            self.splatters[i].forward()
            self.painters[i].forward()
        self.compositor.forward()
        self.mse.forward()

    def backward(self):
        self.mse.backward()
        self.compositor.backward()
        for i in range(len(self.samplers)):
            self.painters[i].backward()
            self.splatters[i].backward()
            self.samplers[i].backward()

    @abstractmethod
    def update(self):
        pass
