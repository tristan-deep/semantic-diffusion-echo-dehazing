import copy
from pathlib import Path

import jax
import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tyro
import zea
from keras import ops
from PIL import Image
from skimage import filters
from zea import Config, init_device, log
from zea.internal.operators import Operator
from zea.models.diffusion import (
    DPS,
    DiffusionModel,
    diffusion_guidance_registry,
)
from zea.tensor_ops import L2
from zea.utils import translate

from plots import create_animation, plot_batch_with_named_masks, plot_dehazed_results
from utils import (
    apply_bottom_preservation,
    extract_skeleton,
    postprocess,
    preprocess,
    smooth_L1,
)


class IdentityOperator(Operator):
    def forward(self, data):
        return data

    def __str__(self):
        return "y = x"


@diffusion_guidance_registry(name="semantic_dps")
class SemanticDPS(DPS):
    def __init__(
        self,
        diffusion_model,
        segmentation_model,
        operator,
        disable_jit=False,
        **kwargs,
    ):
        """Initialize the diffusion guidance.

        Args:
            diffusion_model: The diffusion model to use for guidance.
            operator: The forward (measurement) operator to use for guidance.
            disable_jit: Whether to disable JIT compilation.
        """
        self.diffusion_model = diffusion_model
        self.segmentation_model = segmentation_model
        self.operator = operator
        self.disable_jit = disable_jit
        self.setup(**kwargs)

    def _get_fixed_mask(
        self,
        images,
        bottom_px=40,
        top_px=20,
    ):
        batch_size, height, width, channels = ops.shape(images)

        # Create row indices for each pixel
        row_indices = ops.arange(height)
        row_indices = ops.reshape(row_indices, (height, 1))
        row_indices = ops.tile(row_indices, (1, width))

        # Create top row mask
        fixed_mask = ops.where(
            ops.logical_or(row_indices < top_px, row_indices >= height - bottom_px),
            1.0,
            0.0,
        )
        fixed_mask = ops.expand_dims(fixed_mask, axis=0)
        fixed_mask = ops.expand_dims(fixed_mask, axis=-1)
        fixed_mask = ops.tile(fixed_mask, (batch_size, 1, 1, channels))

        return fixed_mask

    def _get_segmentation_mask(self, images, threshold, sigma):
        input_range = self.diffusion_model.input_range
        images = ops.clip(images, input_range[0], input_range[1])
        images = translate(images, input_range, (-1, 1))

        masks = self.segmentation_model(images)
        mask_vent = masks[..., 0]  # ROI 1 ventricle
        mask_sept = masks[..., 1]  # ROI 2 septum

        def _preprocess_mask(mask):
            mask = ops.convert_to_numpy(mask)
            mask = np.expand_dims(mask, axis=-1)
            mask = np.where(mask > threshold, 1.0, 0.0)
            mask = filters.gaussian(mask, sigma=sigma)
            mask = (mask - ops.min(mask)) / (ops.max(mask) - ops.min(mask) + 1e-8)
            return mask

        mask_vent = _preprocess_mask(mask_vent)
        mask_sept = _preprocess_mask(mask_sept)
        return mask_vent, mask_sept

    def _get_dark_mask(self, images):
        min_val = self.diffusion_model.input_range[0]
        dark_mask = ops.where(ops.abs(images - min_val) < 1e-6, 1.0, 0.0)
        return dark_mask

    def make_omega_map(
        self, images, mask_params, fixed_mask_params, skeleton_params, guidance_kwargs
    ):
        masks = self.get_masks(images, mask_params, fixed_mask_params, skeleton_params)

        masks_vent = masks["vent"]
        masks_sept = masks["sept"]
        masks_fixed = masks["fixed"]
        masks_skeleton = masks["skeleton"]
        masks_dark = masks["dark"]

        masks_strong = ops.clip(
            masks_sept + masks_fixed + masks_skeleton + masks_dark, 0, 1
        )

        background = ops.where(masks_strong < 0.1, 1.0, 0.0) * ops.where(
            masks_vent == 0, 1.0, 0.0
        )

        masks_vent_filtered = masks_vent * (1.0 - masks_strong)

        per_pixel_omega = (
            guidance_kwargs["omega"] * background
            + guidance_kwargs["omega_vent"] * masks_vent_filtered
            + guidance_kwargs["omega_sept"] * masks_strong
        )

        haze_mask_components = (masks_vent > 0.5) * (1 - masks_strong > 0.5)

        haze_mask = []
        for i, m in enumerate(haze_mask_components):
            if scipy.ndimage.label(m)[1] > 1:
                # masks_strong _splits_ masks_vent in 2 or more components
                # so we fall back to masks_vent
                haze_mask.append(masks_vent[i])
                # also remove guidance from this region to avoid bringing haze in
                per_pixel_omega = per_pixel_omega.at[i].set(
                    per_pixel_omega[i] * (1 - masks_vent[i])
                )
            else:
                # masks_strong 'shaves off' some of masks_vent,
                # where there is tissue
                haze_mask.append((masks_vent * (1 - masks_strong))[i])
        haze_mask = ops.stack(haze_mask, axis=0)

        masks["per_pixel_omega"] = per_pixel_omega
        masks["haze"] = haze_mask

        return masks

    def get_masks(self, images, mask_params, fixed_mask_params, skeleton_params):
        """Generate a mask from the input images."""
        masks_vent, masks_sept = self._get_segmentation_mask(images, **mask_params)
        masks_fixed = self._get_fixed_mask(images, **fixed_mask_params)
        masks_skeleton = extract_skeleton(
            images, self.diffusion_model.input_range, **skeleton_params
        )
        masks_dark = self._get_dark_mask(images)
        return {
            "vent": masks_vent,
            "sept": masks_sept,
            "fixed": masks_fixed,
            "skeleton": masks_skeleton,
            "dark": masks_dark,
        }

    def compute_error(
        self,
        noisy_images,
        measurements,
        noise_rates,
        signal_rates,
        per_pixel_omega,
        haze_mask,
        eta=0.01,
        smooth_l1_beta=0.5,
        **kwargs,
    ):
        """Compute measurement error for diffusion posterior sampling.

        Args:
            noisy_images: Noisy images.
            measurement: Target measurement.
            operator: Forward operator.
            noise_rates: Current noise rates.
            signal_rates: Current signal rates.
            omega: Weight for the measurement error.
            omega_mask: Weight for the measurement error at the mask region.
            omega_haze_prior: Weight for the haze prior penalty.
            **kwargs: Additional arguments for the operator.

        Returns:
            Tuple of (measurement_error, (pred_noises, pred_images))
        """
        pred_noises, pred_images = self.diffusion_model.denoise(
            noisy_images,
            noise_rates,
            signal_rates,
            training=False,
        )

        measurement_error = L2(
            per_pixel_omega
            * (measurements - self.operator.forward(pred_images, **kwargs))
        )

        hazy_pixels = pred_images * haze_mask

        # L1 penalty on haze pixels
        # add +1 to make -1 (=black) the 'sparse' value
        haze_prior_error = smooth_L1(hazy_pixels + 1, beta=smooth_l1_beta)

        total_error = measurement_error + eta * haze_prior_error

        return total_error, (pred_noises, pred_images)


def init(config):
    """Initialize models, operator, and guidance objects for semantic-dps dehazing."""

    operator = IdentityOperator()

    diffusion_model = DiffusionModel.from_preset(
        config.diffusion_model_path,
    )
    log.success(
        f"Diffusion model loaded from {log.yellow(config.diffusion_model_path)}"
    )
    segmentation_model = load_segmentation_model(config.segmentation_model_path)

    log.success(
        f"Segmentation model loaded from {log.yellow(config.segmentation_model_path)}"
    )

    guidance_fn = SemanticDPS(
        diffusion_model=diffusion_model,
        segmentation_model=segmentation_model,
        operator=operator,
    )
    diffusion_model._init_operator_and_guidance(operator, guidance_fn)

    return diffusion_model


def load_segmentation_model(path):
    """Load segmentation model"""
    segmentation_model = keras.saving.load_model(path)
    return segmentation_model


def run(
    hazy_images: any,
    diffusion_model: DiffusionModel,
    seed,
    guidance_kwargs: dict,
    mask_params: dict,
    fixed_mask_params: dict,
    skeleton_params: dict,
    batch_size: int = 4,
    diffusion_steps: int = 100,
    initial_diffusion_step: int = 0,
    threshold_output_quantile: float = None,
    preserve_bottom_percent: float = 30.0,
    bottom_transition_width: float = 10.0,
    verbose: bool = True,
):
    input_range = diffusion_model.input_range

    hazy_images = preprocess(hazy_images, normalization_range=input_range)

    pred_tissue_images = []
    masks_out = []
    num_images = hazy_images.shape[0]
    num_batches = (num_images + batch_size - 1) // batch_size

    progbar = keras.utils.Progbar(num_batches, verbose=verbose, unit_name="batch")
    i = 0
    batch_idx = 0
    for i in range(num_batches):
        batch = hazy_images[i * batch_size : (i * batch_size) + batch_size]

        masks = diffusion_model.guidance_fn.make_omega_map(
            batch, mask_params, fixed_mask_params, skeleton_params, guidance_kwargs
        )

        batch_images = diffusion_model.posterior_sample(
            batch,
            n_samples=1,
            n_steps=diffusion_steps,
            initial_step=initial_diffusion_step,
            seed=seed,
            verbose=True,
            per_pixel_omega=masks["per_pixel_omega"],
            haze_mask=masks["haze"],
            eta=guidance_kwargs["eta"],
            smooth_l1_beta=guidance_kwargs["smooth_l1_beta"],
        )
        batch_images = ops.take(batch_images, 0, axis=1)

        pred_tissue_images.append(batch_images)
        masks_out.append(masks)
        batch_idx += 1
        progbar.update(batch_idx)
        i += batch_size

    pred_tissue_images = ops.concatenate(pred_tissue_images, axis=0)
    masks_out = {
        key: ops.concatenate([m[key] for m in masks_out], axis=0)
        for key in masks_out[0].keys()
    }
    pred_haze_images = hazy_images - pred_tissue_images - 1

    if threshold_output_quantile is not None:
        threshold_value = ops.quantile(
            pred_tissue_images, threshold_output_quantile, axis=(1, 2), keepdims=True
        )
        pred_tissue_images = ops.where(
            pred_tissue_images < threshold_value, input_range[0], pred_tissue_images
        )

    # Apply bottom preservation with smooth transition
    if preserve_bottom_percent > 0:
        pred_tissue_images = apply_bottom_preservation(
            pred_tissue_images,
            hazy_images,
            preserve_bottom_percent=preserve_bottom_percent,
            transition_width=bottom_transition_width,
        )

    pred_tissue_images = postprocess(pred_tissue_images, input_range)
    hazy_images = postprocess(hazy_images, input_range)
    pred_haze_images = postprocess(pred_haze_images, input_range)

    return hazy_images, pred_tissue_images, pred_haze_images, masks_out


def main(
    input_folder: str = "./assets",
    output_folder: str = "./temp",
    num_imgs_plot: int = 5,
    device: str = "auto:1",
    config: str = "configs/semantic_dps.yaml",
):
    num_img = num_imgs_plot

    zea.visualize.set_mpl_style()
    init_device(device)

    config = Config.from_yaml(config)
    seed = jax.random.PRNGKey(config.seed)

    paths = list(Path(input_folder).glob("*.png"))
    paths = sorted(paths)

    output_folder = Path(output_folder)

    images = []
    for path in paths:
        image = zea.io_lib.load_image(path)
        images.append(image)
    images = ops.stack(images, axis=0)

    diffusion_model = init(config)

    hazy_images, pred_tissue_images, pred_haze_images, masks = run(
        images,
        diffusion_model=diffusion_model,
        seed=seed,
        **config.params,
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    for image, path in zip(pred_tissue_images, paths):
        image = ops.convert_to_numpy(image)
        file_name = path.name
        Image.fromarray(image).save(output_folder / file_name)

    fig = plot_dehazed_results(
        hazy_images[:num_img],
        pred_tissue_images[:num_img],
        pred_haze_images[:num_img],
        diffusion_model,
        titles=[
            r"Hazy $\mathbf{y}$",
            r"Dehazed $\mathbf{\hat{x}}$",
            r"Haze $\mathbf{\hat{h}}$",
        ],
    )
    path = Path("dehazed_results.png")
    save_kwargs = {"bbox_inches": "tight", "dpi": 300}
    fig.savefig(path, **save_kwargs)
    fig.savefig(path.with_suffix(".pdf"), **save_kwargs)
    log.success(f"Segmentation steps saved to {log.yellow(path)}")

    masks_viz = copy.deepcopy(masks)
    masks_viz.pop("haze")

    num_img = 2  # hardcoded as the plotting figure only neatly supports 2 rows
    masks_viz = {k: v[:num_img] for k, v in masks_viz.items()}

    fig = plot_batch_with_named_masks(
        images[:num_img],
        masks_viz,
        titles=[
            r"Ventricle $v(\mathbf{y})$",
            r"Septum $s(\mathbf{y})$",
            r"Fixed",
            r"Skeleton $t(\mathbf{y})$",
            r"Dark $b(\mathbf{y})$",
            r"Guidance $d(\mathbf{y})$",
        ],
    )
    path = Path("segmentation_steps.png")
    fig.savefig(path, **save_kwargs)
    fig.savefig(path.with_suffix(".pdf"), **save_kwargs)
    log.success(f"Segmentation steps saved to {log.yellow(path)}")

    last_batch_size = len(diffusion_model.track_progress[0])
    create_animation(
        preprocess(hazy_images[-last_batch_size:], diffusion_model.input_range),
        diffusion_model,
        output_path="animation.gif",
        fps=10,
    )

    plt.close("all")


if __name__ == "__main__":
    tyro.cli(main)
