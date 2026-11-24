"""Inference speed benchmark.

Reads all PNG images from the input folder, dehazes them in batches and saves
the results with the original file names. Timing protocol:
- models are loaded and a warm-up batch is run before timing,
- the device is synchronized before and after each timed batch,
- model initialization, disk I/O and output saving are excluded.

The timed region covers the full pipeline per batch: segmentation masks,
diffusion posterior sampling and post-processing.

Usage:
    python benchmark.py --input-folder ./input --output-folder ./output --batch-size 8
"""

import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import numpy as np
import tyro
from keras import ops
from PIL import Image
from zea import Config, init_device, log

from main import init, run
from utils import load_image


def _sync(x):
    """Block until all device work producing ``x`` is done."""
    return jax.block_until_ready(x)


def main(
    input_folder: str = "./input",
    output_folder: str = "./output",
    batch_size: int = 8,
    config: str = "configs/semantic_dps.yaml",
    device: str = "auto:1",
    warmup_batches: int = 1,
    max_batches: int | None = None,
    diffusion_steps: int | None = None,
    jit_loop: bool = True,
):
    init_device(device)
    config = Config.from_path(config)
    params = dict(config.params)
    params["batch_size"] = batch_size
    if diffusion_steps is not None:
        params["diffusion_steps"] = diffusion_steps

    paths = sorted(Path(input_folder).glob("*.png"))
    if not paths:
        log.error(f"No PNG images found in {input_folder}")
        return
    images = np.stack([load_image(p) for p in paths])  # disk I/O, not timed

    diffusion_model = init(config)  # model initialization, not timed
    seed = jax.random.PRNGKey(config.seed)

    batches = [images[i : i + batch_size] for i in range(0, len(images), batch_size)]
    if max_batches is not None:
        batches = batches[:max_batches]

    def infer(batch):
        _, dehazed, _, _ = run(
            batch,
            diffusion_model=diffusion_model,
            seed=seed,
            verbose=False,
            track_progress_type=None,
            jit_sampler=jit_loop,
            **params,
        )
        return _sync(dehazed)  # numpy uint8, (b, h, w)

    for _ in range(warmup_batches):
        infer(batches[0])

    outputs, times = [], []
    for batch in batches:
        _sync(ops.convert_to_tensor(batch))
        t0 = time.perf_counter()
        out = infer(batch)
        times.append(time.perf_counter() - t0)
        outputs.append(out)

    # Only full batches are used for the per-batch statistics.
    full = [t for t, b in zip(times, batches, strict=True) if len(b) == batch_size]
    n_frames = sum(len(b) for b in batches)
    ms_batch = 1e3 * np.mean(full)
    ms_frame = 1e3 * sum(times) / n_frames
    gpu = jax.devices()[0].device_kind

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for img, path in zip(
        np.concatenate(outputs), paths, strict=False
    ):  # output saving, not timed
        Image.fromarray(img).save(output_folder / path.name)

    log.info(f"GPU: {gpu}")
    log.info(
        f"Images: {n_frames}, batch size: {batch_size}, steps: {params['diffusion_steps']}"
    )
    log.info(
        f"Mean inference time per batch: {ms_batch:.1f} ms/batch (+/- {1e3 * np.std(full):.1f})"
    )
    log.info(f"Mean processing time per image: {ms_frame:.1f} ms/frame")
    log.info(f"Throughput: {1e3 / ms_frame:.3f} FPS")


if __name__ == "__main__":
    tyro.cli(main)
