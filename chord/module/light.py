import torch 
from typing import Optional
import torch.nn.functional as Fn
import math
import copy

from . import register
from .base import Base

class BaseLight(Base):
    """
    Base class for light models. 
    """
    
    def setup(self):
        pass

    def forward(self, x: Optional[torch.Tensor] = None):
        """
        Get the light intensity.

        Args:
        x: positions of shape (..., 3). 

        Returns:
        color: radiance intensity of shape (..., 3)
        d: directions of shape (..., 3).
        """
        raise NotImplementedError
    

@register("point-light")
class PointLight(BaseLight):
    """Point light definitions
    """
    def setup(self):
        """Initialize point light.

        Args:
            position (float, float, float): World coordinate of the light.
            color (float, float, float): Light color in (R, G, B).
            power (float): Light power, it will be directly multiplied to each color channel.
        """
        position = self.config.get("position", [0., 0., 10.])
        color = self.config.get("color", [23.47, 21.31, 20.79])
        power = self.config.get("power", 10.) 

        self.register_buffer("position", torch.tensor(position))
        self.register_buffer("color", torch.tensor(color) * power)

    def forward(self, x: Optional[torch.Tensor] = None):
        """Compute light radiance and direction.

        Args:
            x : World coordinate of the interacting surface. [B, H, W, 3]
        Returns:
            color: radiance intensity of shape [B, H, W, 3]
            d: directions of shape [B, H, W, 3], V = (light_pos - world_pos)
        """
        distance    = torch.norm(self.position - x, dim=-1, keepdim=True)
        attenuation = 1.0 / (distance ** 2)
        radiance    = self.color * attenuation
        direction = Fn.normalize(self.position - x, dim=-1)
        return radiance, direction
    
@register("distant-light")
class DistantLight(BaseLight):
    """Distant light definitions
    """
    def setup(self):
        """Initialize distant light.

        Args:
            direction (float, float, float):The direction of light vector.
            color (float, float, float): Light color in (R, G, B).
            power (float): Light power, it will be directly multiplied to each color channel.
        """
        direction = self.config.get("direction", [0., 0., 1.])
        color = self.config.get("color", [23.47, 21.31, 20.79])
        power = self.config.get("power", 0.1)

        self.register_buffer("color", torch.tensor(color) * power)
        self.register_buffer("direction", Fn.normalize(torch.tensor(direction), dim=0))

    def forward(self, x: Optional[torch.Tensor] = None):
        """Compute light radiance and direction.

        Args:
            x : World coordinate of the interacting surface. [B, H, W, 3]
        Returns:
            color: radiance intensity of shape [B, H, W, 3]
            d: directions of shape [B, H, W, 3]
        """
        radiance = self.color.repeat(*x.shape[:-1], 1)
        direction = self.direction.repeat(*x.shape[:-1], 1)
        return radiance, direction