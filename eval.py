import warnings
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import ks_2samp
from zea.io_lib import load_image

import fid_score


def calculate_fid_score(denoised_image_dirs, ground_truth_dir):
    if isinstance(denoised_image_dirs, (str, Path)):
        denoised_image_dirs = [denoised_image_dirs]
    elif not isinstance(denoised_image_dirs, list):
        raise ValueError("Input must be a path or list of paths")

    clean_images_folder = glob(str(ground_truth_dir) + "/*.png")

    print(f"Looking for clean images in: {ground_truth_dir}")
    print(f"Found {len(clean_images_folder)} clean images")

    # Determine optimal batch size based on number of images
    num_denoised = len(denoised_image_dirs)
    num_clean = len(clean_images_folder)
    optimal_batch_size = min(8, num_denoised, num_clean)
    print(f"Using batch size: {optimal_batch_size}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="os.fork.*JAX is multithreaded")

        fid_value = fid_score.calculate_fid_with_cached_ground_truth(
            denoised_image_dirs,
            clean_images_folder,
            batch_size=optimal_batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_workers=2 if torch.cuda.is_available() else 0,
            dims=2048,
        )
    return fid_value


def gcnr(img1, img2):
    """Generalized Contrast-to-Noise Ratio"""
    _, bins = np.histogram(np.concatenate((img1, img2)), bins=256)
    f, _ = np.histogram(img1, bins=bins, density=True)
    g, _ = np.histogram(img2, bins=bins, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))


def cnr(img1, img2):
    """Contrast-to-Noise Ratio"""
    return (img1.mean() - img2.mean()) / np.sqrt(img1.var() + img2.var())


def calculate_cnr_gcnr(result_dehazed_cardiac_ultrasound, mask_path):
    """
    Evaluate gCNR and CNR metrics for denoised images using paired masks.
    Saves detailed and summary statistics to Excel.
    """
    results = []

    mask = np.array(Image.open(mask_path).convert("L"))

    roi1_pixels = result_dehazed_cardiac_ultrasound[mask == 255]  # Foreground ROI
    roi2_pixels = result_dehazed_cardiac_ultrasound[mask == 128]  # Background/Noise ROI

    gcnr_val = gcnr(roi1_pixels, roi2_pixels)
    cnr_val = cnr(roi1_pixels, roi2_pixels)

    results.append([cnr_val, gcnr_val])

    return results


def calculate_ks_statistics(
    result_hazy_cardiac_ultrasound, result_dehazed_cardiac_ultrasound, mask_path
):
    mask = np.array(Image.open(mask_path).convert("L"))

    roi1_original = result_hazy_cardiac_ultrasound[mask == 255]  # region A
    roi1_denoised = result_dehazed_cardiac_ultrasound[mask == 255]
    roi2_original = result_hazy_cardiac_ultrasound[mask == 128]  # region B
    roi2_denoised = result_dehazed_cardiac_ultrasound[mask == 128]

    roi1_ks_stat, roi1_ks_p_value = (None, None)
    roi2_ks_stat, roi2_ks_p_value = (None, None)

    if roi1_original.size > 0 and roi1_denoised.size > 0:
        roi1_ks_stat, roi1_ks_p_value = ks_2samp(roi1_original, roi1_denoised)

    if roi2_original.size > 0 and roi2_denoised.size > 0:
        roi2_ks_stat, roi2_ks_p_value = ks_2samp(roi2_original, roi2_denoised)

    return roi1_ks_stat, roi1_ks_p_value, roi2_ks_stat, roi2_ks_p_value


def calculate_dice_asd(image_path, label_path, checkpoint_path, image_size=224):
    try:
        from test import inference  # Our Segmentation Method
    except ImportError:
        raise ImportError(
            "Segmentation method not available, skipping Dice/ASD calculation"
        )

    pred_img = inference(image_path, checkpoint_path, image_size)
    pred = np.array(pred_img) > 127

    label = Image.open(label_path).convert("L")
    label = label.resize((image_size, image_size), Image.NEAREST)
    label = np.array(label) > 127

    # calculate Dice
    intersection = np.logical_and(pred, label).sum()
    dice = 2 * intersection / (pred.sum() + label.sum() + 1e-8)

    # calculate ASD
    if pred.sum() == 0 or label.sum() == 0:
        asd = np.nan
    else:
        pred_dt = distance_transform_edt(~pred)
        label_dt = distance_transform_edt(~label)

        surface_pred = pred ^ binary_erosion(pred)
        surface_label = label ^ binary_erosion(label)

        d1 = pred_dt[surface_label].mean()
        d2 = label_dt[surface_pred].mean()
        asd = (d1 + d2) / 2

    return dice, asd


def calculate_final_score(aggregates):
    try:
        # (FID + CNR + gCNR):(KS^A + KS^B):(Dice + ASD)= 5:3:2

        group1_score = 0  # FID + CNR + gCNR
        if aggregates.get("fid") is not None:
            fid_min = 60.0
            fid_max = 150.0
            fid_score = (fid_max - aggregates["fid"]) / (fid_max - fid_min)
            fid_score = max(0, min(1, fid_score))
            group1_score += fid_score * 100 * 0.33

        if aggregates.get("cnr_mean") is not None:
            cnr_min = 1.0
            cnr_max = 1.5
            cnr_score = (aggregates["cnr_mean"] - cnr_min) / (cnr_max - cnr_min)
            cnr_score = max(0, min(1, cnr_score))
            group1_score += cnr_score * 100 * 0.33

        if aggregates.get("gcnr_mean") is not None:
            gcnr_min = 0.5
            gcnr_max = 0.8
            gcnr_score = (aggregates["gcnr_mean"] - gcnr_min) / (gcnr_max - gcnr_min)
            gcnr_score = max(0, min(1, gcnr_score))
            group1_score += gcnr_score * 100 * 0.34

        group2_score = 0  # KS^A + KS^B
        if aggregates.get("ks_roi1_ksstatistic_mean") is not None:
            ks1_min = 0.1
            ks1_max = 0.3
            ks1_score = (ks1_max - aggregates["ks_roi1_ksstatistic_mean"]) / (
                ks1_max - ks1_min
            )
            ks1_score = max(0, min(1, ks1_score))
            group2_score += ks1_score * 100 * 0.5

        if aggregates.get("ks_roi2_ksstatistic_mean") is not None:
            ks2_min = 0.0
            ks2_max = 0.5
            ks2_score = (aggregates["ks_roi2_ksstatistic_mean"] - ks2_min) / (
                ks2_max - ks2_min
            )
            ks2_score = max(0, min(1, ks2_score))
            group2_score += ks2_score * 100 * 0.5

        group3_score = 0  # Dice + ASD
        if aggregates.get("dice_mean") is not None:
            dice_min = 0.85
            dice_max = 0.95
            dice_score = (aggregates["dice_mean"] - dice_min) / (dice_max - dice_min)
            dice_score = max(0, min(1, dice_score))
            group3_score += dice_score * 100 * 0.5
        if aggregates.get("asd_mean") is not None:
            asd_min = 0.7
            asd_max = 2.0
            asd_score = (asd_max - aggregates["asd_mean"]) / (asd_max - asd_min)
            asd_score = max(0, min(1, asd_score))
            group3_score += asd_score * 100 * 0.5

        # Final score calculation
        final_score = (group1_score * 5 + group2_score * 3 + group3_score * 2) / 10

        return final_score

    except Exception as e:
        print(f"Error calculating final score: {str(e)}")
        return 0


def plot_metrics(metrics, limits, out_path):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2, 2.7), dpi=600)
    colors = ["#0057b7", "#ffb300", "#008744", "#d62d20"]
    # Arrow direction: ↑ for up, ↓ for down
    metric_labels = {
        "CNR": r"CNR $\uparrow$",
        "gCNR": r"gCNR $\uparrow$",
        "KS_A": r"KS$_{septum}$ $\downarrow$",
        "KS_B": r"KS$_{ventricle}$ $\uparrow$",
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
        ax.set_ylabel("Count", fontsize=10)
        # Draw limits
        if name in limits:
            for lim in limits[name]:
                ax.axvline(lim, color="crimson", linestyle="--", lw=1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=9)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def main(folder: str, roi_folder: str, reference_folder: str):
    folder = Path(folder)
    roi_folder = Path(roi_folder)
    reference_folder = Path(reference_folder)

    folder_files = set(f.name for f in folder.glob("*.png"))
    roi_files = set(f.name for f in roi_folder.glob("*.png"))
    ref_files = set(f.name for f in reference_folder.glob("*.png"))

    print(f"Found {len(folder_files)} .png files in output folder: {folder}")
    print(f"Found {len(roi_files)} .png files in ROI folder: {roi_folder}")
    print(f"Found {len(ref_files)} .png files in reference folder: {reference_folder}")

    # Find intersection of filenames
    common_files = sorted(folder_files & roi_files & ref_files)
    print(f"Found {len(common_files)} images present in all folders.")
    if len(common_files) == 0:
        print("No matching images found in all folders. Check your folder contents.")
        print(f"Output folder files: {sorted(folder_files)}")
        print(f"ROI folder files: {sorted(roi_files)}")
        print(f"Reference folder files: {sorted(ref_files)}")
        assert len(common_files) > 0, (
            "No matching .png files in all folders. Cannot proceed."
        )

    metrics = {"CNR": [], "gCNR": [], "KS_A": [], "KS_B": []}
    limits = {
        "CNR": [1.0, 1.5],
        "gCNR": [0.5, 0.8],
        "KS_A": [0.1, 0.3],
        "KS_B": [0.0, 0.5],
    }

    for name in common_files:
        our_path = folder / name
        roi_path = roi_folder / name
        ref_path = reference_folder / name

        assert our_path.exists(), f"Missing file in output folder: {our_path}"
        assert roi_path.exists(), f"Missing file in ROI folder: {roi_path}"
        assert ref_path.exists(), f"Missing file in reference folder: {ref_path}"

        try:
            img = np.array(load_image(str(our_path)))
            img_ref = np.array(load_image(str(ref_path)))
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            continue

        # CNR/gCNR
        cnr_gcnr = calculate_cnr_gcnr(img, str(roi_path))
        metrics["CNR"].append(cnr_gcnr[0][0])
        metrics["gCNR"].append(cnr_gcnr[0][1])

        # KS statistics
        ks_a, _, ks_b, _ = calculate_ks_statistics(img_ref, img, str(roi_path))
        metrics["KS_A"].append(ks_a)
        metrics["KS_B"].append(ks_b)

    # Compute statistics
    stats = {
        k: (np.mean(v), np.std(v), np.min(v), np.max(v)) for k, v in metrics.items()
    }
    print("Contrast statistics:")
    for k, (mean, std, minv, maxv) in stats.items():
        print(f"{k}: mean={mean:.3f}, std={std:.3f}, min={minv:.3f}, max={maxv:.3f}")

    plot_metrics(metrics, limits, str(folder / "contrast_metrics.png"))
    print(f"Saved metrics plot to {folder / 'contrast_metrics.png'}")

    # Compute FID
    fid_image_paths = [str(folder / name) for name in common_files]
    fid_score = calculate_fid_score(fid_image_paths, str(reference_folder))
    print(f"FID between {folder} and {reference_folder}: {fid_score:.3f}")


if __name__ == "__main__":
    tyro.cli(main)
