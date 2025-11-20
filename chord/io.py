import torch
import imageio.v3 as imageio
import numpy as np
import warnings
import os

import torchvision.transforms.functional as F

def read_image(filename: str, out: torch.Tensor=None) -> torch.Tensor:
    '''
    Read a local image file into a float tensor (pixel values are normalized to [0, 1], CxHxW)

    Args:
        filename: Image file path.
        out: Fill in this tensor rather than return a new tensor if provided.

    Returns:
        Loaded image tensor.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # ignore PIL's user warning that reads fp16 img as fp32
        img: np.ndarray = imageio.imread(filename)

    # Convert the image array to float tensor according to its data type
    res = None
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16 or img.dtype == np.int32:
        img = img.astype(np.float32) / 65535.0
    else:
        raise ValueError(f'Unrecognized image pixel value type: {img.dtype}')
    if img.ndim == 2:
        res = torch.from_numpy(img).unsqueeze(0)  # 1xHxW for grayscale images
    elif img.ndim == 3:
        res = torch.from_numpy(img).movedim(2, 0)[:3] # HxWxC to CxHxW
    else:
        raise ValueError(f'Unrecognized image dimension: {img.shape}')
    
    if out is None:
        return res
    out.copy_(res)

def create_img(img: torch.Tensor):
    '''
    Convert tensor to PIL image

    Args:
        path: Image tensor CxHxW. Squeeze if BxCxHxW and B==1

    Returns:
        PIL image
    '''
    if img.dim() == 4:
        assert img.shape[0] == 1
        img = img.squeeze(0)

    if img.shape[0] == 4:
        out_img = F.to_pil_image(img, mode="CMYK")
        out_img = out_img.convert('RGB')
    elif img.shape[0] == 3:
        out_img = F.to_pil_image(img, mode="RGB")
    elif img.shape[0] == 1:
        out_img = F.to_pil_image(img, mode="L")
    else:
        raise ValueError("Unsupported image dimension.")
    return out_img
    
def save_maps(path: str, maps: dict):
    '''
    Save SVBRDF maps to a given path.

    Args:
        path: Output path.   
        maps: Named maps of tensor images. 
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    for name, image in maps.items():
        out_img = create_img(image)
        out_img.save(os.path.join(path, name+".png"))