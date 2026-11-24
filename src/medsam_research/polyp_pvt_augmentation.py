"""Fixed, opt-in mild augmentation for PVT polyp experiments; no inference use."""
import random

import numpy as np
from PIL import Image, ImageEnhance


POLYP_LIGHT_POLICY = {
    "name": "polyp_light", "version": 1, "stage": "after_resize_before_normalization",
    "horizontal_flip_probability": .5, "vertical_flip_probability": .5,
    "right_angle_rotation_probability": .5, "rotation_choices_degrees": [90, 180, 270],
    "appearance_probability": .5, "brightness_factor": [.9, 1.1],
    "contrast_factor": [.9, 1.1], "saturation_factor": [.95, 1.05],
    "gamma_exponent": [.9, 1.1], "jitter_order": "random", "gamma_after_jitter": True,
    "mask_photometric_changes": False,
}


POLYP_GEOMETRY_POLICY = {
    "name": "polyp_geometry", "version": 1,
    "stage": "after_resize_before_normalization",
    "order": ["crop", "rotation", "horizontal_flip"],
    "random_crop_probability": .5, "crop_side_factor": [.9, 1.0],
    "crop_location": "uniform; no GT-based selection",
    "crop_resize": "bilinear RGB and soft mask, original input size",
    "right_angle_rotation_probability": .5,
    "rotation_choices_degrees": [90, 180, 270],
    "horizontal_flip_probability": .5,
    "vertical_flip_probability": 0., "appearance_probability": 0.,
}


def augmentation_policy(mode):
    if mode not in ("none", "polyp_light", "polyp_geometry"):
        raise ValueError("augmentation must be none, polyp_light or polyp_geometry")
    if mode == "polyp_geometry":
        return dict(POLYP_GEOMETRY_POLICY)
    return dict(POLYP_LIGHT_POLICY) if mode == "polyp_light" else {"name": "none"}


def augment_pair(image, mask, mode="none", rng=None):
    """Exact paired flips/90-degree turns; appearance changes affect RGB only."""
    augmentation_policy(mode)
    if mode == "none":
        return image, mask  # Preserve legacy output and RNG state.
    rng = random if rng is None else rng
    if mode == "polyp_geometry":
        original_size = image.size
        if mask.size != original_size:
            raise ValueError("Image and mask must have matching spatial dimensions")
        if rng.random() < .5:
            width, height = original_size
            factor = rng.uniform(.9, 1.)
            crop_w, crop_h = max(1, round(width * factor)), max(1, round(height * factor))
            left, top = rng.randint(0, width - crop_w), rng.randint(0, height - crop_h)
            box = (left, top, left + crop_w, top + crop_h)
            image = image.crop(box).resize(original_size, Image.Resampling.BILINEAR)
            mask = mask.crop(box).resize(original_size, Image.Resampling.BILINEAR)
        if rng.random() < .5:
            operation = rng.choice((Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                                    Image.Transpose.ROTATE_270))
            image, mask = image.transpose(operation), mask.transpose(operation)
        if rng.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image, mask
    for operation in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
        if rng.random() < .5:
            image, mask = image.transpose(operation), mask.transpose(operation)
    if rng.random() < .5:
        operation = rng.choice((Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                                Image.Transpose.ROTATE_270))
        image, mask = image.transpose(operation), mask.transpose(operation)
    if rng.random() < .5:
        operations = [(ImageEnhance.Brightness, (.9, 1.1)),
                      (ImageEnhance.Contrast, (.9, 1.1)),
                      (ImageEnhance.Color, (.95, 1.05))]
        rng.shuffle(operations)
        for enhancer, limits in operations:
            image = enhancer(image).enhance(rng.uniform(*limits))
        gamma = rng.uniform(.9, 1.1)
        lookup = np.rint(255 * (np.arange(256, dtype=np.float64) / 255) ** gamma).astype(np.uint8)
        image = image.point(lookup.tolist() * len(image.getbands()))
    return image, mask
