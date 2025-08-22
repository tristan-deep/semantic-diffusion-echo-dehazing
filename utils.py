import numpy as np
from keras import ops
from skimage import filters, morphology
from zea.utils import translate


def L1(x):
    """L1 norm of a tensor.

    Implementation of L1 norm: https://mathworld.wolfram.com/L1-Norm.html
    """
    return ops.sum(ops.abs(x))


def smooth_L1(x, beta=0.4):
    """Smooth L1 loss function.

    Implementation of Smooth L1 loss. Large beta values make it similar to L1 loss,
    while small beta values make it similar to L2 loss.
    """
    abs_x = ops.abs(x)
    loss = ops.where(abs_x < beta, 0.5 * x**2 / beta, abs_x - 0.5 * beta)
    return ops.sum(loss)


def postprocess(data, normalization_range):
    """Postprocess data from model output to image."""
    data = ops.clip(data, *normalization_range)
    data = translate(data, normalization_range, (0, 255))
    data = ops.convert_to_numpy(data)
    data = np.squeeze(data, axis=-1)
    return np.clip(data, 0, 255).astype("uint8")


def preprocess(data, normalization_range):
    """Preprocess data for model input. Converts uint8 image(s) in [0, 255] to model input range."""
    data = ops.convert_to_tensor(data, dtype="float32")
    data = translate(data, (0, 255), normalization_range)
    data = ops.expand_dims(data, axis=-1)
    return data


def apply_bottom_preservation(
    output_images, input_images, preserve_bottom_percent=30.0, transition_width=10.0
):
    """Apply bottom preservation with smooth windowed transition.

    Args:
        output_images: Model output images, (batch, height, width, channels)
        input_images: Original input images, (batch, height, width, channels)
        preserve_bottom_percent: Percentage of bottom to preserve from input (default 30%)
        transition_width: Percentage of image height for smooth transition (default 10%)

    Returns:
        Blended images with preserved bottom portion
    """
    output_shape = ops.shape(output_images)

    batch_size, height, width, channels = output_shape

    preserve_height = int(height * preserve_bottom_percent / 100.0)
    transition_height = int(height * transition_width / 100.0)

    transition_start = height - preserve_height - transition_height
    preserve_start = height - preserve_height

    transition_start = max(0, transition_start)
    preserve_start = min(height, preserve_start)

    if transition_start >= preserve_start:
        transition_start = preserve_start
        transition_height = 0

    y_coords = ops.arange(height, dtype="float32")
    y_coords = ops.reshape(y_coords, (height, 1, 1))

    if transition_height > 0:
        # Smooth transition using cosine interpolation
        transition_region = ops.logical_and(
            y_coords >= transition_start, y_coords < preserve_start
        )

        transition_progress = (y_coords - transition_start) / transition_height
        transition_progress = ops.clip(transition_progress, 0.0, 1.0)

        # Use cosine for smooth transition (0.5 * (1 - cos(π * t)))
        cosine_weight = 0.5 * (1.0 - ops.cos(np.pi * transition_progress))

        blend_weight = ops.where(
            y_coords < transition_start,
            0.0,
            ops.where(
                transition_region,
                cosine_weight,
                1.0,
            ),
        )
    else:
        # No transition, just hard switch
        blend_weight = ops.where(y_coords >= preserve_start, 1.0, 0.0)

    blend_weight = ops.expand_dims(blend_weight, axis=0)

    blended_images = (1.0 - blend_weight) * output_images + blend_weight * input_images

    return blended_images


def extract_skeleton(images, input_range, sigma_pre=4, sigma_post=4, threshold=0.3):
    """Extract skeletons from the input images."""
    images_np = ops.convert_to_numpy(images)
    images_np = np.clip(images_np, input_range[0], input_range[1])
    images_np = translate(images_np, input_range, (0, 1))
    images_np = np.squeeze(images_np, axis=-1)

    skeleton_masks = []
    for img in images_np:
        img[img < threshold] = 0
        smoothed = filters.gaussian(img, sigma=sigma_pre)
        binary = smoothed > filters.threshold_otsu(smoothed)
        skeleton = morphology.skeletonize(binary)
        skeleton = morphology.dilation(skeleton, morphology.disk(2))
        skeleton = filters.gaussian(skeleton.astype(np.float32), sigma=sigma_post)
        skeleton_masks.append(skeleton)

    skeleton_masks = np.array(skeleton_masks)
    skeleton_masks = np.expand_dims(skeleton_masks, axis=-1)

    # normalize to [0, 1]
    min_val, max_val = np.min(skeleton_masks), np.max(skeleton_masks)
    skeleton_masks = (skeleton_masks - min_val) / (max_val - min_val + 1e-8)

    return ops.convert_to_tensor(skeleton_masks, dtype=images.dtype)
