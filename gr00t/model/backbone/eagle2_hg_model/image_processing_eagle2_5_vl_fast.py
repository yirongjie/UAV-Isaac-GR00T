# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# <--- MODIFIED: Nagdagdag ng 'Any', 'Dict', 'Tuple' para sa Python 3.8
from typing import List, Optional, Union, Any, Dict, Tuple
from functools import partial

# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/image_processing_llava_onevision_fast.py

from transformers.image_processing_utils import (
    BatchFeature,
    # <--- MODIFIED: Inalis ang 'get_patch_output_size' (na nagdudulot ng ImportError)
)
from transformers.image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    # <--- MODIFIED: Inalis ang 'DefaultFastImageProcessorKwargs', 'group_images_by_shape', 'reorder_images'
)
from transformers.image_utils import IMAGENET_STANDARD_MEAN  # 0.5, 0.5, 0.5
from transformers.image_utils import IMAGENET_STANDARD_STD  # 0.5, 0.5, 0.5
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    # <--- MODIFIED: Inalis ang 'VideoInput', 'make_flat_list_of_images', 'validate_kwargs'
    get_image_size,
)
# <--- MODIFIED: Inalis ang 'Unpack'
from transformers.utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    # <--- MODIFIED: Inalis ang 'is_torchvision_v2_available'
)

if is_torch_available():
    import torch

# <--- MODIFIED: Inalis ang 'is_torchvision_v2_available' check at 'pil_torch_interpolation_mapping'
#               Direktang ginamit ang 'torchvision.transforms.functional'
from torchvision.transforms import functional as F


# <--- MODIFIED: Idinagdag ang 'make_flat_list_of_images' na wala sa v4.38.2
def make_flat_list_of_images(images: ImageInput) -> List[ImageInput]:
    """
    Makes a flat list of images from a nested list of images or a single image.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)):
        return [item for sublist in images for item in sublist]
    if isinstance(images, (list, tuple)):
        return images
    return [images]


# <--- MODIFIED: Idinagdag ang 'group_images_by_shape' na wala sa v4.38.2
def group_images_by_shape(
    images: List["torch.Tensor"],
) -> Tuple[Dict[Tuple[int, int], "torch.Tensor"], List[Tuple[int, int]]]:
    """
    Groups images by their shape.
    """
    grouped_images = {}
    grouped_images_index = []
    for image in images:
        shape = tuple(image.shape[1:])  # (C, H, W) -> (H, W)
        if shape not in grouped_images:
            grouped_images[shape] = []
        grouped_images[shape].append(image)
        grouped_images_index.append(shape)

    for shape, image_list in grouped_images.items():
        grouped_images[shape] = torch.stack(image_list, dim=0)

    return grouped_images, grouped_images_index


# <--- MODIFIED: Idinagdag ang 'reorder_images' na wala sa v4.38.2
def reorder_images(
    grouped_images: Dict[Tuple[int, int], "torch.Tensor"],
    grouped_images_index: List[Tuple[int, int]],
) -> List["torch.Tensor"]:
    """
    Reorders images based on the original index.
    """
    images = []
    image_counters = {shape: 0 for shape in grouped_images.keys()}
    for shape in grouped_images_index:
        images.append(grouped_images[shape][image_counters[shape]])
        image_counters[shape] += 1
    return images


# <--- MODIFIED: Backported 'get_patch_output_size' (nawawala sa transformers 4.38.2)
# Ito ang function na nagdulot ng iyong ImportError.
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


# <--- MODIFIED: Inalis ang 'DefaultFastImageProcessorKwargs'
class Eagle2_5_VLFastImageProcessorKwargs:
    max_dynamic_tiles: Optional[int]
    min_dynamic_tiles: Optional[int]
    use_thumbnail: Optional[bool]
    pad_during_tiling: Optional[bool]
    do_pad: Optional[bool]


@add_start_docstrings(
    "Constructs a fast ConvNeXT image processor. Based on [`SiglipImageProcessor`] with incorporation of processing each video frame.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        image_grid_pinpoints (`List[List[int]]`, *optional*):
            A list of possible resolutions to use for processing high resolution images. The best resolution is selected
            based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
            method. Not used for processing videos.
        do_pad (`bool`, *optional*):
            Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
            number of patches in the batch. Padding will be applied to the bottom and right with zeros.
    """,
)
class Eagle2_5_VLImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 448, "width": 448}
    default_to_square = False
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    max_dynamic_tiles = 12
    min_dynamic_tiles = 1
    use_thumbnail = True
    pad_during_tiling = False
    # <--- MODIFIED: Inalis ang 'valid_kwargs = Eagle2_5_VLFastImageProcessorKwargs'
    model_input_names = ["pixel_values_videos"]

    # <--- MODIFIED: Inalis ang 'Unpack' sa signature
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
            max_dynamic_tiles (`int`, *optional*):
                The maximum number of dynamic tiles to use for processing high resolution images.
            min_dynamic_tiles (`int`, *optional*):
                The minimum number of dynamic tiles to use for processing high resolution images.
            use_thumbnail (`bool`, *optional*):
                Whether to use a thumbnail for processing high resolution images.
            pad_during_tiling (`bool`, *optional*):
                Whether to pad the image during tiling.
            do_pad (`bool`, *optional*):
                    Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                    number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        """,
    )

    # NOTE(YL): we will overload the preprocess method to add the image_flags
    # def preprocess(
    #     self, images: ImageInput, **kwargs: Unpack[Eagle2_5_VLFastImageProcessorKwargs]
    # ) -> BatchFeature:
    #     return super().preprocess(images, **kwargs)

    def _prepare_images_structure(
        self,
        images: ImageInput,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        # <--- MODIFIED: Ginamit ang backported function
        return make_flat_list_of_images(images)

    # <--- MODIFIED: Pinalitan ang 'VideoInput' ng 'ImageInput'
    def _prepare_videos_structure(self, videos: ImageInput) -> ImageInput:
        return self._prepare_images_structure(videos)

    def _prepare_input_videos(
        self,
        # <--- MODIFIED: Pinalitan ang 'VideoInput' ng 'ImageInput'
        videos: ImageInput,
        do_convert_rgb: Optional[bool] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # <--- MODIFIED: Pinalitan ang 'list["torch.Tensor"]' ng 'List["torch.Tensor"]'
        device: Optional["torch.device"] = None,
    ) -> List["torch.Tensor"]:
        """
        Prepare the input images for processing.
        """
        videos = self._prepare_videos_structure(videos)
        process_video_fn = partial(
            self._process_image,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )
        # todo: yoni - check if we can parallelize this efficiently
        processed_videos = []
        for video in videos:
            processed_videos.append(process_video_fn(video))

        return processed_videos

    def _resize_for_patching(
        self,
        image: "torch.Tensor",
        # <--- MODIFIED: Pinalitan ang 'tuple' ng 'Tuple'
        target_resolution: Tuple[int, int],
        interpolation: "F.InterpolationMode",
        input_data_format: ChannelDimension,
    ) -> "torch.Tensor":
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image ("torch.Tensor"):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            interpolation (`InterpolationMode`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            "torch.Tensor": The resized and padded image.
        """
        # <--- MODIFIED: Tumatawag na ngayon sa backported function na 'get_patch_output_size'
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)

        # Resize the image
        resized_image = F.resize(image, (new_height, new_width), interpolation=interpolation)

        return resized_image

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

    def _pad_for_patching(
        self, image: "torch.Tensor", 
        # <--- MODIFIED: Pinalitan ang 'tuple' ng 'Tuple'
        target_resolution: Tuple[int, int], 
        input_data_format: ChannelDimension
    ) -> "torch.Tensor":
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution
        # <--- MODIFIED: Tumatawag na ngayon sa backported function na 'get_patch_output_size'
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        padded_image = F.pad(image, padding=[paste_x, paste_y, paste_x, paste_y])

        return padded_image

    def _get_image_patches(
        self,
        image: "torch.Tensor",
        min_num: int,
        max_num: int,
        # <--- MODIFIED: Pinalitan ang 'tuple' ng 'Tuple'
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

    def _pad_for_batching(
        self,
        pixel_values: List["torch.Tensor"],
    ) -> List["torch.Tensor"]:
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`List[torch.Tensor]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)

        Returns:
            List[`torch.Tensor`]: The padded images.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            torch.nn.functional.pad(image, pad=[0, 0, 0, 0, 0, 0, 0, max_patch - image.shape[0]])
            for image in pixel_values
        ]

        return pixel_values

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        max_dynamic_tiles: int,
        min_dynamic_tiles: int,
        use_thumbnail: bool,
        pad_during_tiling: bool,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        do_pad: bool,
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:
        processed_images = []
        image_sizes = []
        # Determine the size tuple
        if size and size.height and size.width:
            size_tuple = (size.height, size.width)
        else:
            size_tuple = (size.shortest_edge, size.shortest_edge)

        # Determine the patch size
        if crop_size and crop_size.height:
            tile_size = crop_size.height
        elif size and size.height:
            tile_size = size.height
        else:
            tile_size = size.shortest_edge

        for image in images:
            image_patches = self._get_image_patches(
                image,
                min_num=min_dynamic_tiles,
                max_num=max_dynamic_tiles,
                size=size_tuple,
                tile_size=tile_size,
                use_thumbnail=use_thumbnail,
                interpolation=interpolation,
                pad_during_tiling=pad_during_tiling,
            )

            # Group images by size for batched processing
            processed_image_patches_grouped = {}
            # <--- MODIFIED: Ginamit ang backported function
            grouped_image_patches, grouped_image_patches_index = group_images_by_shape(
                image_patches
            )

            for shape, stacked_image_patches in grouped_image_patches.items():
                if do_resize:
                    stacked_image_patches = self.resize(
                        image=stacked_image_patches,
                        size=size,
                        interpolation=interpolation,
                    )
                if do_center_crop:
                    stacked_image_patches = self.center_crop(stacked_image_patches, crop_size)
                
                # <--- MODIFIED: Pinalitan ang 'rescale_and_normalize' ng 'rescale' at 'normalize'
                # Fused rescale and normalize
                if do_rescale:
                    stacked_image_patches = self.rescale(
                        stacked_image_patches, rescale_factor
                    )
                if do_normalize:
                    stacked_image_patches = self.normalize(
                        stacked_image_patches, image_mean, image_std
                    )
                
                processed_image_patches_grouped[shape] = stacked_image_patches
            
            # <--- MODIFIED: Ginamit ang backported function
            processed_image_patches = reorder_images(
                processed_image_patches_grouped, grouped_image_patches_index
            )
            processed_image_patches = (
                torch.stack(processed_image_patches, dim=0)
                if return_tensors
                else processed_image_patches
            )
            processed_images.append(processed_image_patches)
            image_sizes.append(get_image_size(image, ChannelDimension.FIRST))

        if do_pad:
            processed_images = self._pad_for_batching(processed_images)

        # processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        processed_images = (
            torch.cat(processed_images, dim=0) if return_tensors else processed_images
        )
        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes},
            tensor_type=return_tensors,
        )

    def preprocess(
        self,
        images: ImageInput,
        # <--- MODIFIED: Pinalitan ang 'VideoInput' ng 'ImageInput' at inalis ang 'Unpack'
        videos: ImageInput = None,
        **kwargs,
    ) -> BatchFeature:
        # <--- MODIFIED: Inalis ang 'validate_kwargs'
        
        # <--- MODIFIED: Manwal na 'kwargs.setdefault' loop
        # Get base kwargs from the base class
        base_kwargs_names = [
            "do_resize", "size", "resample", "do_center_crop", "crop_size", "do_rescale",
            "rescale_factor", "do_normalize", "image_mean", "image_std", "do_convert_rgb",
            "default_to_square", "data_format", "input_data_format", "device", "return_tensors"
        ]
        # Add custom kwargs
        custom_kwargs_names = [
            "max_dynamic_tiles", "min_dynamic_tiles", "use_thumbnail", "pad_during_tiling", "do_pad"
        ]
        for kwarg_name in base_kwargs_names + custom_kwargs_names:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")
        # Prepare input images
        if images is not None:
            images = self._prepare_input_images(
                images=images,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        if videos is not None:
            videos = self._prepare_input_images(
                images=videos,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # torch resize uses interpolation instead of resample
        resample = kwargs.pop("resample")
        # <--- MODIFIED: Inalis ang 'pil_torch_interpolation_mapping' at ginamit ang 'F.InterpolationMode(resample.value)'
        kwargs["interpolation"] = (
            F.InterpolationMode(resample.value)
            if isinstance(resample, (PILImageResampling, int))
            else resample
        )

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")
        if images is not None:
            return self._preprocess(images, **kwargs)
        elif videos is not None:
            return self._preprocess(videos, **kwargs)


__all__ = ["Eagle2_5_VLImageProcessorFast"]