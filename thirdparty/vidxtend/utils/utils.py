from __future__ import annotations
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchvision

from einops import rearrange
from PIL import Image

from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "blur",
    "dilate_erode",
    "gaussian_blur",
    "get_frame_rate",
    "human_size",
    "latent_friendly_image",
    "rectify_image",
    "reiterator",
    "scale_image",
    "debug_tensors"
]

def human_size(num_bytes: int) -> str:
    """
    Convert a number of bytes to a human-readable string
    """
    for unit in ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} YB"

def latent_friendly_image(
    image: Union[List[Image.Image], Image.Image],
    nearest: int=8,
    resample=Image.NEAREST
) -> Union[List[Image.Image], Image.Image]:
    """
    Resize an image or list of images to be friendly to latent space optimization
    """
    if isinstance(image, list):
        return [latent_friendly_image(img, nearest) for img in image]
    width, height = image.size
    new_width = (width // nearest) * nearest
    new_height = (height // nearest) * nearest
    image = image.resize((new_width, new_height), resample=resample)
    return image

def scale_image(
    image: Union[List[Image.Image], Image.Image],
    scale: float=1.0,
    resample=Image.LANCZOS
) -> Union[List[Image.Image], Image.Image]:
    """
    Scale an image or list of images
    """
    if scale == 1.0:
        return image
    if isinstance(image, list):
        return [scale_image(img, scale) for img in image]
    width, height = image.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), resample=resample)
    return image

def dilate_erode(
    image: Union[Image, List[Image]],
    value: int
) -> Union[Image, List[Image]]:
    """
    Given an image, dilate or erode it.
    Values of >0 dilate, <0 erode. 0 Does nothing.
    :see: http://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    """
    if value == 0:
        return image
    if isinstance(image, list):
        return [
            dilate_erode(img, value)
            for img in image
        ]

    from PIL import Image
    import cv2
    import numpy as np

    arr = np.array(image.convert("L"))
    transform = cv2.dilate if value > 0 else cv2.erode
    value = abs(value)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    arr = transform(arr, kernel, iterations=1)
    return Image.fromarray(arr)

def blur(
    image: Union[Image, List[Image]],
    kernel_size: int
) -> Union[Image, List[Image]]:
    """
    Given an image, blur it.
    """
    if kernel_size == 0:
        return image
    if isinstance(image, list):
        return [
            blur(img, kernel_size)
            for img in image
        ]

    from PIL import Image
    import cv2
    import numpy as np
    arr = np.array(image)
    arr = cv2.blur(arr, (kernel_size, kernel_size))
    return Image.fromarray(arr)

def gaussian_blur(
    image: Union[Image, List[Image]],
    kernel_size: int
) -> Union[Image, List[Image]]:
    """
    Given an image, blur it with a Gaussian kernel.
    """
    if kernel_size == 0:
        return image
    if isinstance(image, list):
        return [
            gaussian_blur(img, kernel_size)
            for img in image
        ]

    from PIL import Image
    import cv2
    import numpy as np
    arr = np.array(image)
    arr = cv2.GaussianBlur(arr, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
    return Image.fromarray(arr)

def rectify_image(
    image: Union[List[Image.Image], Image.Image],
    size: int,
    resample=Image.LANCZOS,
    method: Literal["smallest", "largest"]="largest"
) -> Union[List[Image.Image], Image.Image]:
    """
    Scale an image or list of images
    """
    if isinstance(image, list):
        return [rectify_image(img, size) for img in image]

    width, height = image.size

    if width > height and method == "largest":
        new_width = size
        new_height = int(size * height / width)
    else:
        new_width = int(size * width / height)
        new_height = size
    image = image.resize((new_width, new_height), resample=resample)
    return image

class reiterator:
    """
    Transparently memoize any iterator
    """
    memoized: List[Any]

    def __init__(self, iterable: Iterable[Any]) -> None:
        self.iterable = iterable
        self.memoized = []
        self.started = False
        self.finished = False

    def __iter__(self) -> Iterable[Any]:
        if not self.started:
            self.started = True
            last_index: Optional[int] = None
            for i, value in enumerate(self.iterable):
                yield value
                self.memoized.append(value)
                last_index = i
                if self.finished:
                    # Completed somewhere else
                    break
            if self.finished:
                if last_index is None:
                    last_index = 0
                for value in self.memoized[last_index+1:]:
                    yield value
            self.finished = True
            del self.iterable
        elif not self.finished:
            # Complete iterator
            self.memoized += [item for item in self.iterable]
            self.finished = True
            del self.iterable
            for item in self.memoized:
                yield item
        else:
            for item in self.memoized:
                yield item

def get_frame_rate(video_path: str) -> float:
    """
    Returns the frame rate of the given video file using moviepy.
    
    Parameters:
    video_path (str): The path to the video file.
    
    Returns:
    float: The frame rate of the video.
    """
    from moviepy.editor import VideoFileClip
    with VideoFileClip(video_path) as video:
        return video.fps  # fps stands for frames per second

def debug_tensors(*args: Any, **kwargs: Any) -> None:
    """
    Logs tensors
    """
    import torch
    from vidxtend.utils.log_utils import logger
    include_bounds = kwargs.pop("include_bounds", False)
    arg_dict = dict([
        (f"arg_{i}", arg)
        for i, arg in enumerate(args)
    ])
    for tensor_dict in [arg_dict, kwargs]:
        for key, value in tensor_dict.items():
            if isinstance(value, list) or isinstance(value, tuple):
                for i, v in enumerate(value):
                    debug_tensors(include_bounds=include_bounds, **{f"{key}_{i}": v})
            elif isinstance(value, dict):
                for k, v in value.items():
                    debug_tensors(include_bounds=include_bounds, **{f"{key}_{k}": v})
            elif isinstance(value, torch.Tensor):
                if include_bounds:
                    t_min, t_max = value.aminmax()
                    logger.debug(f"{key} = {value.shape} ({value.dtype}) on {value.device}, min={t_min}, max={t_max}")
                else:
                    logger.debug(f"{key} = {value.shape} ({value.dtype}) on {value.device}")
