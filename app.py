import os

import gradio as gr
import jax
import numpy as np
import spaces
from PIL import Image

from main import Config, init, run
from utils import load_image

CONFIG_PATH = "configs/semantic_dps.yaml"
SLIDER_CONFIG_PATH = "configs/slider_params.yaml"
ASSETS_DIR = "assets"

description = """
# Semantic Diffusion Posterior Sampling for Cardiac Ultrasound Dehazing
Select an example image below. The algorithm will dehaze the image. Note that the algorithm was heavily tuned for the DehazingEcho2025 challenge dataset, and not optimized for generalization. Therefore it is not expected to work well on any type of echocardiogram.

Two parameters that are interesting to control and adjust the amount of dehazing are the "Omega (Ventricle)" and "Eta (haze prior)"
"""


def initialize_model():
    config = Config.from_yaml(CONFIG_PATH)
    diffusion_model = init(config)
    return config, diffusion_model


@spaces.GPU(duration=30)

# Generator function for status updates
def process_image(input_img, diffusion_steps, omega, omega_vent, omega_sept, eta):
    if input_img is None:
        yield (
            gr.update(
                value='<div style="background:#ffeeba;padding:10px;border-radius:6px;font-weight:bold;font-size:1.1em;color:#856404;">⚠️ No input image was provided. Please select or upload an image before running.</div>'
            ),
            None,
        )
        return
    # Show loading message
    yield (
        gr.update(
            value='<div style="background:#ffeeba;padding:10px;border-radius:6px;font-weight:bold;font-size:1.1em;color:#856404;">⏳ Loading model...</div>'
        ),
        None,
    )

    try:
        config, diffusion_model = initialize_model()
        params = config.params
    except Exception as e:
        yield (
            gr.update(
                value=f'<div style="background:#f8d7da;padding:10px;border-radius:6px;font-weight:bold;font-size:1.1em;color:#721c24;">❌ Error initializing model: {e}</div>'
            ),
            None,
        )
        return

    def _prepare_image(image):
        resized = False
        if image.mode != "L":
            image = image.convert("L")
        orig_shape = image.size[::-1]
        h, w = diffusion_model.input_shape[:2]
        if image.size != (w, h):
            image = image.resize((w, h), Image.BILINEAR)
            resized = True
        image = np.array(image)
        image = image.astype(np.float32)
        image = image[None, ...]
        return image, resized, orig_shape

    try:
        image, resized, orig_shape = _prepare_image(input_img)
    except Exception as e:
        yield (
            gr.update(
                value=f'<div style="background:#f8d7da;padding:10px;border-radius:6px;font-weight:bold;font-size:1.1em;color:#721c24;">❌ Error preparing input image: {e}</div>'
            ),
            None,
        )
        return

    guidance_kwargs = {
        "omega": omega,
        "omega_vent": omega_vent,
        "omega_sept": omega_sept,
        "eta": eta,
        "smooth_l1_beta": params["guidance_kwargs"]["smooth_l1_beta"],
    }

    seed = jax.random.PRNGKey(config.seed)

    try:
        yield (
            gr.update(
                value='<div style="background:#ffeeba;padding:10px;border-radius:6px;font-weight:bold;font-size:1.1em;color:#856404;">🌀 Running dehazing algorithm... (First time takes longer...) <span style="font-weight:normal;font-size:0.95em;">(first time takes longer)</span></div>'
            ),
            None,
        )
        _, pred_tissue_images, *_ = run(
            hazy_images=image,
            diffusion_model=diffusion_model,
            seed=seed,
            guidance_kwargs=guidance_kwargs,
            mask_params=params["mask_params"],
            fixed_mask_params=params["fixed_mask_params"],
            skeleton_params=params["skeleton_params"],
            batch_size=1,
            diffusion_steps=diffusion_steps,
            threshold_output_quantile=params.get("threshold_output_quantile", None),
            preserve_bottom_percent=params.get("preserve_bottom_percent", 30.0),
            bottom_transition_width=params.get("bottom_transition_width", 10.0),
            verbose=False,
        )
    except Exception as e:
        yield (
            gr.update(
                value=f'<div style="background:#f8d7da;padding:10px;border-radius:6px;font-weight:bold;font-size:1.1em;color:#721c24;">❌ The algorithm failed to process the image: {e}</div>'
            ),
            None,
        )
        return

    out_img = np.squeeze(pred_tissue_images[0])
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)
    out_pil = Image.fromarray(out_img)
    if resized and out_pil.size != (orig_shape[1], orig_shape[0]):
        out_pil = out_pil.resize((orig_shape[1], orig_shape[0]), Image.BILINEAR)
    yield gr.update(value="Done!"), (input_img, out_pil)
    yield (
        gr.update(
            value='<div style="background:#d4edda;padding:10px;border-radius:6px;font-weight:bold;font-size:1.1em;color:#155724;">✅ Done!</div>'
        ),
        (input_img, out_pil),
    )


slider_params = Config.from_yaml(SLIDER_CONFIG_PATH)

diffusion_steps_default = slider_params["diffusion_steps"]["default"]
diffusion_steps_min = slider_params["diffusion_steps"]["min"]
diffusion_steps_max = slider_params["diffusion_steps"]["max"]
diffusion_steps_step = slider_params["diffusion_steps"]["step"]

omega_default = slider_params["omega"]["default"]
omega_min = slider_params["omega"]["min"]
omega_max = slider_params["omega"]["max"]
omega_step = slider_params["omega"]["step"]

omega_vent_default = slider_params["omega_vent"]["default"]
omega_vent_min = slider_params["omega_vent"]["min"]
omega_vent_max = slider_params["omega_vent"]["max"]
omega_vent_step = slider_params["omega_vent"]["step"]

omega_sept_default = slider_params["omega_sept"]["default"]
omega_sept_min = slider_params["omega_sept"]["min"]
omega_sept_max = slider_params["omega_sept"]["max"]
omega_sept_step = slider_params["omega_sept"]["step"]

eta_default = slider_params["eta"]["default"]
eta_min = slider_params["eta"]["min"]
eta_max = slider_params["eta"]["max"]
eta_step = slider_params["eta"]["step"]


example_image_paths = [
    os.path.join(ASSETS_DIR, f)
    for f in os.listdir(ASSETS_DIR)
    if f.lower().endswith(".png")
]
example_images = [load_image(p) for p in example_image_paths]
examples = [[img] for img in example_images]


with gr.Blocks() as demo:
    gr.Markdown(description)
    status = gr.Markdown("", visible=True)
    with gr.Row():
        img1 = gr.Image(label="Input Image", type="pil", webcam_options=False)
        img2 = gr.ImageSlider(label="Dehazed Image", type="pil")
    gr.Examples(examples=examples, inputs=[img1])
    with gr.Row():
        diffusion_steps_slider = gr.Slider(
            minimum=diffusion_steps_min,
            maximum=diffusion_steps_max,
            step=diffusion_steps_step,
            value=diffusion_steps_default,
            label="Diffusion Steps",
        )
        omega_slider = gr.Slider(
            minimum=omega_min,
            maximum=omega_max,
            step=omega_step,
            value=omega_default,
            label="Omega (background)",
        )
        omega_vent_slider = gr.Slider(
            minimum=omega_vent_min,
            maximum=omega_vent_max,
            step=omega_vent_step,
            value=omega_vent_default,
            label="Omega Ventricle",
        )
        omega_sept_slider = gr.Slider(
            minimum=omega_sept_min,
            maximum=omega_sept_max,
            step=omega_sept_step,
            value=omega_sept_default,
            label="Omega Septum",
        )
        eta_slider = gr.Slider(
            minimum=eta_min,
            maximum=eta_max,
            step=eta_step,
            value=eta_default,
            label="Eta (haze prior)",
        )
    run_btn = gr.Button("Run")

    run_btn.click(
        process_image,
        inputs=[
            img1,
            diffusion_steps_slider,
            omega_slider,
            omega_vent_slider,
            omega_sept_slider,
            eta_slider,
        ],
        outputs=[status, img2],
        queue=True,
    )

if __name__ == "__main__":
    demo.launch()
