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


def augmentation_policy(mode):
    if mode not in ("none", "polyp_light"):
        raise ValueError("augmentation must be none or polyp_light")
    return dict(POLYP_LIGHT_POLICY) if mode == "polyp_light" else {"name": "none"}


def augment_pair(image, mask, mode="none", rng=None):
    """Exact paired flips/90-degree turns; appearance changes affect RGB only."""
    augmentation_policy(mode)
    if mode == "none":
        return image, mask  # Preserve legacy output and RNG state.
    rng = random if rng is None else rng
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
