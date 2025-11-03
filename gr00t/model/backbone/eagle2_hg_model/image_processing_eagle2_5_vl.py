# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
#
# MODIFIED FOR transformers==4.38.2, pytorch==2.1.0, python==3.8
# --------------------------------------------------------

from typing import List, Optional, Union, Any, Dict, Tuple
from functools import partial

import torch
from torchvision.transforms import functional as F
from PIL import Image

from transformers.image_utils import is_torch_tensor

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    # <--- MODIFIED: 移除了 'SizeDict'，因为它在 v4.38.2 中不存在
    get_image_size,
)
from transformers.utils import TensorType, add_start_docstrings, is_torch_available, logging


logger = logging.get_logger(__name__)


# <--- MODIFIED: 回填(Backport) 'make_flat_list_of_images' (在 v4.38.2 中缺失)
def make_flat_list_of_images(images: ImageInput) -> List[ImageInput]:
    """
    Makes a flat list of images from a nested list of images or a single image.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)):
        return [item for sublist in images for item in sublist]
    if isinstance(images, (list, tuple)):
        return images
    return [images]


# <--- MODIFIED: 回填(Backport) 'get_patch_output_size' (在 v4.38.2 中缺失)
def get_patch_output_size(
    image: ImageInput,
    target_resolution: Tuple[int, int],
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    Utility function to get the output size of an image after resizing it to a target resolution
    while maintaining the aspect ratio.
    """
    original_height, original_width = get_image_size(image, channel_dim=input_data_format)
    target_height, target_width = target_resolution

    # Determine the new size preserving aspect ratio
    if original_width > original_height:
        new_width = target_width
        new_height = int(target_width * original_height / original_width)
    else:
        new_height = target_height
        new_width = int(target_height * original_width / original_height)

    # Ensure new dimensions are at least 1
    new_height = max(1, new_height)
    new_width = max(1, new_width)

    return new_height, new_width


# <--- MODIFIED: 保留 'crop' 函数，因为它处理 Tensor
def crop(img: torch.Tensor, left: int, top: int, right: int, bottom: int) -> torch.Tensor:
    """Crop the given numpy array.

    Args:
        img (torch.Tensor): Image to be cropped. Format should be (C, H, W).
        left (int): The left coordinate of the crop box.
        top (int): The top coordinate of the crop box.
        right (int): The right coordinate of the crop box.
        bottom (int): The bottom coordinate of the crop box.

    Returns:
        torch.Tensor: Cropped image.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("img should be torch.Tensor. Got {}".format(type(img)))

    if img.ndim not in [2, 3]:
        raise ValueError("Image should have 2 or 3 dimensions. Got {}".format(img.ndim))

    img_height = img.shape[1]
    img_width = img.shape[2]
    if top < 0 or left < 0 or bottom > img_height or right > img_width:
        raise ValueError("Crop coordinates out of bounds")

    if top >= bottom or left >= right:
        raise ValueError("Invalid crop coordinates")

    return img[:, top:bottom, left:right]


# <--- MODIFIED: 更改为继承 'BaseImageProcessor' (非 "Fast" 版本)
class Eagle2_5_VLImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Eagle2_5_VL image processor.
    This processor is a "slow" implementation based on PIL and TorchVision,
    adapted from the "fast" tensor-based logic.
    It is compatible with transformers==4.38.2.
    """
    model_input_names = ["pixel_values_videos"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        do_pad: bool = True,
        max_dynamic_tiles: int = 12,
        min_dynamic_tiles: int = 1,
        use_thumbnail: bool = True,
        pad_during_tiling: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 448, "width": 448}
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_convert_rgb = do_convert_rgb

        # Custom attributes from the "fast" version
        self.do_pad = do_pad
        self.max_dynamic_tiles = max_dynamic_tiles
        self.min_dynamic_tiles = min_dynamic_tiles
        self.use_thumbnail = use_thumbnail
        self.pad_during_tiling = pad_during_tiling

    # <--- MODIFIED: 为 transformers 4.38.2 添加缺失的 'to_tensor' 辅助方法
    def to_tensor(self, array) -> "torch.Tensor":
        """
        将一个 Numpy 数组转换为 PyTorch Tensor。
        (这是一个简化的回填版本，假设 array 是 HWC 格式的 np.ndarray)
        """
        if not is_torch_available():
            raise ImportError("PyTorch is required to convert numpy arrays to tensors.")
        
        # 'to_tensor' 在旧版本 torchvision 中是 HWC -> CHW
        # 但 transformers 的基类希望我们只转换类型
        # 我们需要模拟 'BaseImageProcessor.to_tensor' 的行为
        
        # 1. 确保它是 torch tensor
        tensor = torch.tensor(array)

        # 2. 如果是 HWC (3维)，则转换为 CHW
        if tensor.ndim == 3 and tensor.shape[-1] in [1, 3, 4]:
            tensor = tensor.permute(2, 0, 1)
            
        return tensor

    # <--- MODIFIED: 为 transformers 4.38.2 添加缺失的 'rescale' 辅助方法
    def rescale(self, image, scale: float) -> "torch.Tensor":
        """
        按一个因子缩放图像张量。
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image to rescale must be a torch.Tensor.")
            
        return image * scale

    # <--- MODIFIED: 为 transformers 4.38.2 添加缺失的 'normalize' 辅助方法
    def normalize(self, image, mean, std) -> "torch.Tensor":
        """
        标准化一个图像张量。
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image to normalize must be a torch.Tensor.")

        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=image.dtype, device=image.device)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=image.dtype, device=image.device)

        # 确保 mean/std 可以广播 (例如，从 [3] 变为 [3, 1, 1])
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        return (image - mean) / std

    def to_numpy_array(self, image) -> "np.ndarray":
        """
        将一个 PIL 图像转换为 Numpy 数组。
        (这是一个简化的回填版本，假设 image 是 PIL 图像)
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("Numpy is required to convert PIL images to arrays.")

        if hasattr(image, "convert"): # 检查它是否像 PIL 图像
            # 假设 'image' 已经是 PIL Image 对象
            return np.array(image)
        
        # 作为后备，如果它已经是 numpy 数组
        if isinstance(image, np.ndarray):
            return image

        raise ValueError(f"Unsupported image type for to_numpy_array: {type(image)}")

    # <--- MODIFIED: 为 transformers 4.38.2 添加缺失的 'convert_rgb' 辅助方法
    def convert_rgb(self, image) -> "Image.Image":
        """
        将一个 PIL 图像转换为 RGB 格式。
        """
        # 假设此时 'image' 已经是 PIL Image 对象（由 convert_to_pil 保证）
        if image.mode == "RGB":
            return image
        return image.convert("RGB")
    
    # <--- MODIFIED: 为 transformers 4.38.2 最终修正 'convert_to_pil' (修复 numpy 检查)
    def convert_to_pil(self, images: ImageInput) -> List[Image.Image]:
        """
        将单个图像或一批图像转换为 PIL 图像列表。
        """
        # --- 修正导入 ---
        # 我们只需要 is_torch_tensor
        from transformers.utils import is_torch_tensor
        from PIL import Image

        # --- 修正Numpy检查 ---
        # 直接在函数内部尝试导入 numpy，而不是依赖 'is_numpy_available'
        try:
            import numpy as np
            _NUMPY_AVAILABLE = True
        except ImportError:
            _NUMPY_AVAILABLE = False
        # --- 结束修正 ---

        if not isinstance(images, list):
            images = [images]

        # 如果所有图像都已经是 PIL 图像，则直接返回列表
        if all(isinstance(image, Image.Image) for image in images):
            return images

        # 转换为 PIL 图像
        if is_torch_tensor(images[0]):
            images = [self.to_pil_image(image) for image in images]
        # --- 修正调用 ---
        # 使用我们自己定义的 _NUMPY_AVAILABLE 标志
        elif _NUMPY_AVAILABLE and isinstance(images[0], np.ndarray):
            images = [self.to_pil_image(image) for image in images]
        # --- 结束修正 ---
        elif isinstance(images[0], Image.Image):
            pass
        else:
            raise ValueError(f"Unsupported image type: {type(images[0])}")
        
        return images
    # <--- MODIFIED: 保留所有核心算法 (find_closest_aspect_ratio)
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """
        previous version mainly foucs on ratio.
        We also consider area ratio here.
        """
        best_factor = float("-inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            # ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            # area_ratio = (ratio[0] * ratio[1] * image_size * image_size) / area
            """
            new area > 60% of original image area is enough.
            """
            factor_based_on_area_n_ratio = min(
                (ratio[0] * ratio[1] * image_size * image_size) / area, 0.6
            ) * min(target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio)

            if factor_based_on_area_n_ratio > best_factor:
                best_factor = factor_based_on_area_n_ratio
                best_ratio = ratio

        return best_ratio

    # <--- MODIFIED: 保留所有核心算法 (_resize_for_patching)
    def _resize_for_patching(
        self,
        image: "torch.Tensor",
        target_resolution: Tuple[int, int],
        interpolation: "F.InterpolationMode",
        input_data_format: ChannelDimension,
    ) -> "torch.Tensor":
        """
        Resizes an image to a target resolution while maintaining aspect ratio.
        """
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)
        resized_image = F.resize(image, (new_height, new_width), interpolation=interpolation)
        return resized_image

    # <--- MODIFIED: 保留所有核心算法 (_pad_for_patching)
    def _pad_for_patching(
        self, image: "torch.Tensor", target_resolution: Tuple[int, int], input_data_format: ChannelDimension
    ) -> "torch.Tensor":
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        padded_image = F.pad(image, padding=[paste_x, paste_y, paste_x, paste_y])
        return padded_image

    # <--- MODIFIED: 保留所有核心算法 (_get_image_patches)
    def _get_image_patches(
        self,
        image: "torch.Tensor",
        min_num: int,
        max_num: int,
        size: Tuple[int, int],
        tile_size: int,
        use_thumbnail: bool,
        interpolation: "F.InterpolationMode",
        pad_during_tiling: bool,
    ) -> List["torch.Tensor"]:
        image_size = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        orig_height, orig_width = image_size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, tile_size
        )

        # calculate the target width and height
        target_width = tile_size * target_aspect_ratio[0]
        target_height = tile_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        if pad_during_tiling:
            resized_image = self._resize_for_patching(
                image,
                (target_height, target_width),
                interpolation=interpolation,
                input_data_format=ChannelDimension.FIRST,
            )
            padded_image = self._pad_for_patching(
                resized_image,
                (target_height, target_width),
                input_data_format=ChannelDimension.FIRST,
            )
            image_used_to_split = padded_image
        else:
            image_used_to_split = F.resize(
                image, (target_height, target_width), interpolation=interpolation
            )

        processed_tiles = []
        for i in range(blocks):
            box = (
                (i % (target_width // tile_size)) * tile_size,
                (i // (target_width // tile_size)) * tile_size,
                ((i % (target_width // tile_size)) + 1) * tile_size,
                ((i // (target_width // tile_size)) + 1) * tile_size,
            )
            # split the image
            split_img = crop(image_used_to_split, box[0], box[1], box[2], box[3])
            processed_tiles.append(split_img)
        assert len(processed_tiles) == blocks

        if use_thumbnail and len(processed_tiles) != 1:
            thumbnail_img = F.resize(image, (tile_size, tile_size), interpolation=interpolation)
            processed_tiles.append(thumbnail_img)

        return processed_tiles

    # <--- MODIFIED: 保留所有核心算法 (_pad_for_batching)
    def _pad_for_batching(
        self,
        pixel_values: List["torch.Tensor"],
    ) -> List["torch.Tensor"]:
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            torch.nn.functional.pad(image, pad=[0, 0, 0, 0, 0, 0, 0, max_patch - image.shape[0]])
            for image in pixel_values
        ]

        return pixel_values

    # <--- MODIFIED: 这是一个新的辅助函数，用于封装 "Fast" 逻辑
    def _preprocess_patches(
        self,
        images: List["torch.Tensor"],
        max_dynamic_tiles: int,
        min_dynamic_tiles: int,
        use_thumbnail: bool,
        pad_during_tiling: bool,
        # <--- MODIFIED: 签名从 SizeDict 更改为 Dict[str, int]
        size: Dict[str, int],
        crop_size: Dict[str, int],
        interpolation: "F.InterpolationMode",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
        do_pad: bool,
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:
        """
        Applies the tensor-based patch logic (tiling, rescaling, normalizing).
        """
        processed_images = []
        image_sizes = []
        
        # Determine the size tuple
        # <--- MODIFIED: 使用 .get() 方法安全访问字典
        if size and size.get("height") and size.get("width"):
            size_tuple = (size["height"], size["width"])
        else:
            # 假设 'shortest_edge' 存在，如果 'height'/'width' 不存在的话
            size_tuple = (size.get("shortest_edge"), size.get("shortest_edge"))

        # Determine the patch size
        if crop_size and crop_size.get("height"):
            tile_size = crop_size["height"]
        elif size and size.get("height"):
            tile_size = size["height"]
        else:
            tile_size = size.get("shortest_edge")
        
        # 修复当 size 为 None 时可能出现的 NoneType 错误
        if tile_size is None:
             raise ValueError("Could not determine tile size. 'size' or 'crop_size' must be provided.")

        for image in images: # `images` is a List[torch.Tensor]
            image_patches = self._get_image_patches(
                image,
                min_num=min_dynamic_tiles,
                max_num=max_dynamic_tiles,
                size=size_tuple,
                tile_size=tile_size,
                use_thumbnail=use_thumbnail,
                interpolation=interpolation,
                pad_during_tiling=pad_during_tiling,
            ) # This returns List[torch.Tensor]

            # Stack them into one tensor for batch processing: (num_patches, C, H, W)
            stacked_image_patches = torch.stack(image_patches, dim=0)

            # Apply rescale and normalize (these BaseImageProcessor methods work on Tensors)
            if do_rescale:
                stacked_image_patches = self.rescale(stacked_image_patches, rescale_factor)
            if do_normalize:
                stacked_image_patches = self.normalize(stacked_image_patches, image_mean, image_std)

            processed_images.append(stacked_image_patches)
            image_sizes.append(get_image_size(image, ChannelDimension.FIRST))

        if do_pad:
            processed_images = self._pad_for_batching(processed_images)

        if return_tensors and len(processed_images) > 0:
            processed_images = torch.cat(processed_images, dim=0)
        
        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes},
            tensor_type=return_tensors,
        )

    # <--- MODIFIED: 重写 'preprocess' 方法以使用 "Slow" 基类和 "Fast" 算法
    def preprocess(
        self,
        images: ImageInput,
        videos: Optional[ImageInput] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        do_pad: Optional[bool] = None,
        max_dynamic_tiles: Optional[int] = None,
        min_dynamic_tiles: Optional[int] = None,
        use_thumbnail: Optional[bool] = None,
        pad_during_tiling: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        
        # 1. Handle kwargs
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        return_tensors = return_tensors if return_tensors is not None else self.return_tensors
        
        # Custom kwargs
        do_pad = do_pad if do_pad is not None else self.do_pad
        max_dynamic_tiles = max_dynamic_tiles if max_dynamic_tiles is not None else self.max_dynamic_tiles
        min_dynamic_tiles = min_dynamic_tiles if min_dynamic_tiles is not None else self.min_dynamic_tiles
        use_thumbnail = use_thumbnail if use_thumbnail is not None else self.use_thumbnail
        pad_during_tiling = pad_during_tiling if pad_during_tiling is not None else self.pad_during_tiling
        
        if "data_format" in kwargs:
            kwargs.pop("data_format")
        
        # 2. Combine images and videos
        if images is None and videos is None:
            raise ValueError("You must provide either 'images' or 'videos'.")
        
        all_media = []
        if images is not None:
            all_media.extend(make_flat_list_of_images(images))
        if videos is not None:
            logger.warning("The 'videos' argument is being treated as a list of images.")
            all_media.extend(make_flat_list_of_images(videos))
            
        # 3. Use "Slow" processor logic to get PIL Images
        # _prepare_input_images is a protected method, but we need it.
        # It handles conversion to PIL.
        images_pil = self.convert_to_pil(all_media)

        # 4. Manually convert PIL to Tensors (the "Hybrid" step)
        # We skip the base class's resize/crop, as our patch logic handles it.
        tensor_images = []
        for img in images_pil:
            if do_convert_rgb:
                img = self.convert_rgb(img)
            # Convert to numpy, then to tensor (standard BaseImageProcessor flow)
            img_array = self.to_numpy_array(img)
            tensor_images.append(self.to_tensor(img_array))
        
        # 5. Handle interpolation mode for torchvision
        # interpolation = F.InterpolationMode(resample.value) if isinstance(resample, (PILImageResampling, int)) else resample
        # <--- MODIFIED: 修复 resample 逻辑，正确处理 int 和 Enum
        if isinstance(resample, PILImageResampling):
            resample_value = resample.value
        elif isinstance(resample, int):
            resample_value = resample
        else:
            # 假定它已经是正确的 'F.InterpolationMode' 成员（不太可能，但作为后备）
            interpolation = resample
            resample_value = -1 # 标记为已处理

        # 2. 将整数值映射到 torchvision v2 需要的枚举成员
        if resample_value != -1:
            if resample_value == 0:   # PIL.Image.NEAREST
                interpolation = F.InterpolationMode.NEAREST
            elif resample_value == 2: # PIL.Image.BILINEAR
                interpolation = F.InterpolationMode.BILINEAR
            elif resample_value == 3: # PIL.Image.BICUBIC
                interpolation = F.InterpolationMode.BICUBIC
            elif resample_value == 1: # PIL.Image.LANCZOS
                interpolation = F.InterpolationMode.LANCZOS
            elif resample_value == 4: # PIL.Image.BOX
                interpolation = F.InterpolationMode.BOX
            elif resample_value == 5: # PIL.Image.HAMMING
                interpolation = F.InterpolationMode.HAMMING
            else:
                # 如果整数无效，则抛出错误
                raise ValueError(f"{resample_value} 不是一个有效的 PIL 重采样整数")
        # --- 结束修正 ---

        # 6. Call the patch-based preprocessing
        return self._preprocess_patches(
            images=tensor_images,
            max_dynamic_tiles=max_dynamic_tiles,
            min_dynamic_tiles=min_dynamic_tiles,
            use_thumbnail=use_thumbnail,
            pad_during_tiling=pad_during_tiling,
            size=size,
            crop_size=crop_size,
            interpolation=interpolation,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            return_tensors=return_tensors,
        )

__all__ = ["Eagle2_5_VLImageProcessor"]