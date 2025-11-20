import torch

def vector_dot(A: torch.Tensor, B: torch.Tensor, min=0.0) -> torch.Tensor:
    return torch.clamp((A * B).sum(1, keepdim=True), min=min, max=1.0)

def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4)).to(f.dtype)
    
def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055).to(f.dtype)

def tone_gamma(x: torch.Tensor) -> torch.Tensor:
    x = 1 - torch.exp(-x)
    return torch.pow(x, 1.0/2.2)

# safe division for value range 0-1
class safe_01_div(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return torch.div(a, torch.clamp(b, min=1e-4, max=1.0))

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_input = grad_output.clone()

        return torch.div(1, torch.clamp(b, min=1e-4, max=1.0)) * grad_input, -1 * torch.div(a, torch.clamp(b, min=1e-2, max=1.0)**2) * grad_input


def get_positions(h, w, real_size, use_pixel_centers=True) -> torch.Tensor:
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = torch.meshgrid(
        torch.arange(h) + pixel_center,
        torch.arange(w) + pixel_center,
        indexing='ij'
    )
    if not isinstance(real_size, list):
        real_size = [real_size] * 2
    pos = torch.stack([(i / h - 0.5) * real_size[0], (j / w - 0.5) * real_size[1], torch.zeros_like(i)], dim=-1)
    return pos
    
# N, H: （Bx3xHxW), roughness: (Bx1xHxW)
# The "D", facet distribution function in Cook-Torrence model
def DistributionGGX(cosNH, roughness):
    a = roughness * roughness
    a2 = a * a
    cosNH2 = cosNH * cosNH
    num = a2
    denom = cosNH2 * (a2 - 1.0) + 1.0
    denom = torch.pi * denom * denom
    return num / denom

# NdotV, roughness: (Bx1xHxW)
def GeometrySchlickGGX(NdotV: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    r = (roughness + 1.0)
    k = (r*r) / 8.0

    num   = NdotV
    denom = NdotV * (1.0 - k) + k
	
    return num / denom

# cosTheta, F0 (Bx1xHxW)
# The "F"
def fresnelSchlick(cosTheta: torch.Tensor, F0: torch.Tensor) -> torch.Tensor:
    return F0 + (1.0 - F0) * torch.pow(1.0 - cosTheta, 5.0)