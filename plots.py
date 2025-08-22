import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import keras
import matplotlib.pyplot as plt
import numpy as np
import tyro
from keras import ops
from matplotlib.patches import PathPatch
from matplotlib.path import Path as pltPath
from PIL import Image
from skimage import measure
from zea import log
from zea.utils import save_to_gif
from zea.visualize import plot_image_grid

from utils import postprocess


def add_shape_from_mask(ax, mask, **kwargs):
    """add a shape to axis from mask array.

    Args:
        ax (plt.ax): matplotlib axis
        mask (ndarray): numpy array with non-zero
            shape defining the region of interest.
    Kwargs:
        edgecolor (str): color of the shape's edge
        facecolor (str): color of the shape's face
        linewidth (int): width of the shape's edge

    Returns:
        plt.ax: matplotlib axis with shape added
    """
    # Pad mask to ensure edge contours are found
    padded_mask = np.pad(mask, pad_width=1, mode="constant", constant_values=0)
    contours = measure.find_contours(padded_mask, 0.5)
    patches = []
    for contour in contours:
        # Remove padding offset
        contour -= 1
        path = pltPath(contour[:, ::-1])
        patch = PathPatch(path, **kwargs)
        patches.append(ax.add_patch(patch))
    return patches


def matplotlib_figure_to_numpy(fig):
    """Convert matplotlib figure to numpy array.

    Args:
        fig (matplotlib.figure.Figure): figure to convert.

    Returns:
        np.ndarray: numpy array of figure.

    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image = np.array(image)[..., :3]
    buf.close()
    return image


def plot_batch_with_named_masks(
    images, masks_dict, mask_colors=None, titles=None, **kwargs
):
    """
    Plot batch of images in rows, each column overlays a different mask from the dict.
    Mask labels are shown as column titles. If mask name is 'per_pixel_omega', show it
    directly with inferno colormap (no overlay).

    Args:
        images: np.ndarray, shape (batch, height, width, channels)
        masks_dict: dict of {name: mask}, each mask shape  (batch, height, width, channels)
        mask_colors: dict of {name: color} or None (default colors used)
    """
    mask_names = list(masks_dict.keys())
    batch_size = images.shape[0]
    default_colors = ["red", "green", "#33aaff", "yellow", "magenta", "cyan"]
    mask_colors = mask_colors or {
        name: default_colors[i % len(default_colors)]
        for i, name in enumerate(mask_names)
    }

    # Prepare images for each column
    columns = []
    cmaps = []
    for name in mask_names:
        if name == "per_pixel_omega":
            mask_np = np.array(masks_dict[name])
            columns.append(np.squeeze(mask_np))
            cmaps.append(["inferno"] * batch_size)
        else:
            columns.append(np.squeeze(images))
            cmaps.append(["gray"] * batch_size)

    # Stack columns: shape (num_columns, batch, ...)
    all_images = np.stack(columns, axis=0)  # (num_columns, batch, ...)
    # Rearrange to (batch, num_columns, ...)
    all_images = (
        np.transpose(all_images, (1, 0, 2, 3, 4))
        if all_images.ndim == 5
        else np.transpose(all_images, (1, 0, 2, 3))
    )
    # Flatten to (batch * num_columns, ...)
    all_images = all_images.reshape(batch_size * len(mask_names), *images.shape[1:])

    # Flatten cmaps for plot_image_grid in the same order as images
    flat_cmaps = []
    for row in range(batch_size):
        for col in range(len(mask_names)):
            flat_cmaps.append(cmaps[col][row])

    fig, _ = plot_image_grid(
        all_images,
        ncols=len(mask_names),
        remove_axis=False,
        cmap=flat_cmaps,
        figsize=(8, 3.3),
        **kwargs,
    )

    # Overlay masks for non-per_pixel_omega columns
    for col_idx, name in enumerate(mask_names):
        if name == "per_pixel_omega":
            continue
        mask_np = np.array(masks_dict[name])
        axes = fig.axes[col_idx : batch_size * len(mask_names) : len(mask_names)]
        for ax, mask_img in zip(axes, mask_np):
            add_shape_from_mask(
                ax, mask_img.squeeze(), color=mask_colors[name], alpha=0.3
            )

    # Add column titles
    row_idx = 0
    if titles is None:
        titles = mask_names
    for col_idx, name in enumerate(titles):
        ax_idx = row_idx * len(mask_names) + col_idx
        fig.axes[ax_idx].set_title(name, fontsize=9, color="white")
        fig.axes[ax_idx].set_facecolor("black")

    # Add colorbar for per_pixel_omega if present
    if "per_pixel_omega" in mask_names:
        col_idx = mask_names.index("per_pixel_omega")
        axes = fig.axes[col_idx : batch_size * len(mask_names) : len(mask_names)]

        # Get vertical bounds of the subplot column
        top_ax = axes[0]
        bottom_ax = axes[-1]
        top_pos = top_ax.get_position()
        bottom_pos = bottom_ax.get_position()

        full_y0 = bottom_pos.y0
        full_y1 = top_pos.y1
        full_height = full_y1 - full_y0

        # Manually shrink to 80% of full height and center vertically
        scale = 0.8
        height = full_height * scale
        y0 = full_y0 + (full_height - height) / 2

        x0 = top_pos.x1 + 0.015  # Horizontal position to the right
        width = 0.015  # Thin bar

        # Add colorbar axis
        cax = fig.add_axes([x0, y0, width, height])

        im = axes[0].get_images()[0] if axes[0].get_images() else None
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"Guidance weighting $\mathbf{p}$")
        cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        cbar.ax.yaxis.set_tick_params(labelsize=7)
        cbar.ax.yaxis.label.set_size(8)

    return fig


def plot_dehazed_results(
    hazy_images,
    pred_tissue_images,
    pred_haze_images,
    diffusion_model,
    titles=("Hazy", "Dehazed", "Haze"),
):
    """Create and save visualization with optional mask overlays."""

    # Create the processed image stack using the helper function
    input_shape = diffusion_model.input_shape
    stack_images = ops.stack(
        [
            hazy_images,
            pred_tissue_images,
            pred_haze_images,
        ]
    )
    stack_images = ops.reshape(stack_images, (-1, input_shape[0], input_shape[1]))

    # Define labels based on what we're showing
    fig, _ = plot_image_grid(
        stack_images,
        ncols=len(hazy_images),
        remove_axis=False,
        vmin=0,
        vmax=255,
    )
    # Set labels and styling
    for i, ax in enumerate(fig.axes):
        if i % len(hazy_images) == 0:
            label = titles[(i // len(hazy_images)) % len(titles)]
            ax.set_ylabel(label, fontsize=12)

    return fig


def plot_metrics(metrics, limits, out_path):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2, 2.7), dpi=200)
    colors = ["#0057b7", "#ffb300", "#008744", "#d62d20"]
    metric_labels = {
        "CNR": r"CNR $\uparrow$",
        "gCNR": r"gCNR $\uparrow$",
        "KS_A": r"KS$_{septum}$ $\downarrow$",
        "KS_B": r"KS$_{ventricle}$ $\uparrow$",
    }
    # For legend handles
    legend_handles = []
    import matplotlib.lines as mlines

    min_style = {
        "color": "crimson",
        "linestyle": "--",
        "lw": 1.2,
        "marker": "o",
        "markersize": 5,
    }
    max_style = {
        "color": "crimson",
        "linestyle": ":",
        "lw": 1.2,
        "marker": "s",
        "markersize": 5,
    }
    for idx, (ax, (name, values)) in enumerate(zip(axes, metrics.items())):
        ax.hist(
            values,
            bins=30,
            color=colors[idx % len(colors)],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.7,
        )
        ax.set_xlabel(metric_labels.get(name, name), fontsize=11)
        if idx == 0:
            ax.set_ylabel("Count", fontsize=10)
        # Draw limits and collect legend handles only once
        if name in limits:
            lims = limits[name]
            if len(legend_handles) == 0:
                # Only add legend handles for the first metric
                min_handle = mlines.Line2D([], [], **min_style, label="min score")
                max_handle = mlines.Line2D([], [], **max_style, label="max score")
                legend_handles.extend([min_handle, max_handle])
            if len(lims) > 0:
                ax.axvline(lims[0], **min_style)
            if len(lims) > 1:
                ax.axvline(lims[1], **max_style)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=9)
    # Place legend above all subplots
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    return fig


def plot_optimization_history_from_json(
    trials_data: List[Dict[str, Any]], output_path: Path, method: str
):
    """Plot optimization history directly from JSON data."""

    # Extract completed trials with values
    completed_trials = [
        t for t in trials_data if t["state"] == "COMPLETE" and t["value"] is not None
    ]

    if not completed_trials:
        print("No completed trials found!")
        return

    # Sort by trial number
    completed_trials.sort(key=lambda x: x["number"])

    trial_numbers = [t["number"] for t in completed_trials]
    values = [t["value"] for t in completed_trials]

    # Apply seaborn styling
    plt.style.use("seaborn-v0_8-darkgrid")

    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 3), dpi=600)

    # Plot all trial values with styling similar to plots.py
    ax.scatter(
        trial_numbers,
        values,
        c="#0057b7",
        alpha=0.6,
        s=30,
        edgecolor="black",
        linewidth=0.5,
    )

    # Plot best value so far
    best_values = []
    current_best = values[0]
    for val in values:
        if val > current_best:  # Assuming maximization
            current_best = val
        best_values.append(current_best)

    ax.plot(
        trial_numbers,
        best_values,
        color="#d62d20",
        linewidth=2.5,
        label="Best Value",
        marker="o",
        markersize=4,
        markevery=max(1, len(trial_numbers) // 20),
    )

    ax.set_xlabel("Trial", fontsize=11)
    ax.set_ylabel("Objective Value", fontsize=11)
    # ax.set_title("Optimization History", fontsize=12)
    ax.legend(fontsize=10, frameon=False)

    # Remove top and right spines like in plots.py
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=9)

    # Save plot
    fig.savefig(
        output_path / f"optimization_history_{method}.png", dpi=600, bbox_inches="tight"
    )
    fig.savefig(
        output_path / f"optimization_history_{method}.pdf", dpi=600, bbox_inches="tight"
    )
    plt.close(fig)


def create_animation_frame(hazy_images, tissue_frame, haze_frame):
    """Create a single animation frame from the tracked progress."""
    batch, height, width = ops.shape(hazy_images)
    frame_stack = ops.stack(
        [
            hazy_images,
            tissue_frame,
            haze_frame,
        ]
    )
    frame_stack = ops.reshape(frame_stack, (-1, height, width))
    fig_frame, _ = plot_image_grid(
        frame_stack,
        ncols=len(hazy_images),
        remove_axis=False,
        vmin=0,
        vmax=255,
    )
    labels = ["Hazy", "Haze", "Tissue"]
    for i, ax in enumerate(fig_frame.axes):
        label = labels[i % len(labels)]
        ax.set_ylabel(label, fontsize=12)
    frame_array = matplotlib_figure_to_numpy(fig_frame)
    plt.close(fig_frame)
    return frame_array


def create_animation(hazy_images, diffusion_model, output_path, fps):
    """Create animation from tracked progress frames."""
    if not (len(diffusion_model.track_progress) > 1):
        log.warning(
            "Animation requested but no intermediate frames were tracked. "
            "Try reducing diffusion_steps or ensure progress tracking is enabled."
        )
        return

    log.info(f"Creating animation with {len(diffusion_model.track_progress)} frames...")

    animation_frames = []
    progbar = keras.utils.Progbar(
        len(diffusion_model.track_progress), unit_name="frame"
    )
    for tissue_frame in diffusion_model.track_progress:
        haze_frame = hazy_images - tissue_frame - 1
        tissue_frame = postprocess(tissue_frame, diffusion_model.input_range)
        haze_frame = postprocess(haze_frame, diffusion_model.input_range)
        _hazy_images = postprocess(hazy_images, diffusion_model.input_range)
        frame_array = create_animation_frame(_hazy_images, tissue_frame, haze_frame)
        animation_frames.append(frame_array)
        progbar.add(1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    animation_path = Path(output_path).with_suffix(".gif")
    save_to_gif(animation_frames, animation_path, fps=fps)


def main(json_file: str, output_dir: str = "plots", method: str = "semantic_dps"):
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    # Load trial data
    with open(json_path, "r") as f:
        trials_data = json.load(f)

    print(f"Loaded {len(trials_data)} trials from {json_file}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating optimization history plot...")
    plot_optimization_history_from_json(trials_data, output_path, method)


if __name__ == "__main__":
    tyro.cli(main)
