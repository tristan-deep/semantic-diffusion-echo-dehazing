<div align="center">
	<h1>Semantic Diffusion Posterior Sampling for Cardiac Ultrasound Dehazing</h1>
	<p>
		<a href="https://github.com/tristan-deep/semantic-diffusion-echo-dehazing">
			<img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white" alt="GitHub">
		</a>
		<a href="https://arxiv.org/abs/2508.17326">
			<img src="https://img.shields.io/badge/arXiv-B31B1B?style=flat&logo=arXiv&logoColor=white" alt="Paper">
		</a>
		<a href="https://huggingface.co/collections/tristan-deep/semantic-diffusion-posterior-sampling-for-cardiac-ultrasound-68a70559a7f719c7e6bd5788">
			<img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000" alt="Hugging Face">
		</a>
		<a href="https://keras.io/"><img src="https://img.shields.io/badge/Keras-EE4C2C?logo=keras&logoColor=white" alt="Keras"></a>
	</p>
		<h3>
			<a href="https://tristan-deep.github.io/">Tristan Stevens</a> &nbsp;|&nbsp;
			<a href="https://oisinnolan.github.io/">Oisín Nolan</a> &nbsp;|&nbsp;
			<a href="https://www.tue.nl/en/research/researchers/ruud-van-sloun">Ruud van Sloun</a>
		</h3>
	<p>Eindhoven University of Technology, the Netherlands</p>
</div>

<p align="center">
	<img src="https://github.com/tristan-deep/semantic-diffusion-echo-dehazing/raw/main/paper/animation.gif" alt="Cardiac Ultrasound Dehazing Animation" style="max-width: 100%; height: auto;">
</p>

> [!NOTE]
> An improved method that exploits the temporal context of echocardiography sequences is available: [Nuclear Diffusion Models for Low-Rank Background Suppression in Videos](https://doi.org/10.1109/ICASSP55912.2026.11463877) (ICASSP 2026). See the [code demo](https://zea.readthedocs.io/en/latest/notebooks/models/nuclear_dehazing_example.html).

### Installation

The algorithm is implemented using Keras with JAX backend. Furthermore it heavily relies on the [zea ultrasound library](https://github.com/tue-bmd/zea).

Either install the following in your Python environment, or use the [Dockerfile](./Dockerfile) provided in this repository.

```bash
# requires Python>=3.10 environment
pip install -r requirements.txt
```

Also install [JAX](https://github.com/google/jax#installation). Note this can vary depending on your system.

```bash
pip install jax[cuda12]
```

> [!NOTE]
> Although the code was primarily tested with JAX as the Keras backend, TensorFlow and PyTorch should also work.

### Running the algorithm

Some example images are downloaded in the [./assets](./assets) folder. The models are automatically downloaded from the [Hugging Face Model Hub](https://huggingface.co/collections/tristan-deep/semantic-diffusion-posterior-sampling-for-cardiac-ultrasound-68a70559a7f719c7e6bd5788).

```bash
python main.py --input-folder ./assets --output-folder ./temp
```

You can also dehaze frames directly from the [DehazingEcho2025 dataset](https://huggingface.co/datasets/zeahub/DehazingEcho2025) (or any local [zea](https://github.com/tue-bmd/zea) file):

```bash
python main.py --input-folder hf://zeahub/DehazingEcho2025/noisy/patient-1-4C.hdf5 --frames 0 1 2 --output-folder ./temp
```

Alternatively, you can use the Gradio app provided in this repository to interact with the model via a web interface. To launch the app, run:

```bash
python app.py
```

<p align="center">
	<img src="https://github.com/tristan-deep/semantic-diffusion-echo-dehazing/raw/main/paper/gradio_demo.gif" alt="Gradio Demo" style="max-width: 100%; height: auto;">
</p>

### Inference speed benchmark

Put 256 × 256 PNG images in `./input` and run:

```bash
python benchmark.py --input-folder ./input --output-folder ./output --batch-size 8
```

The models are loaded and warmed up first, then each batch is timed (excluding disk I/O). The dehazed images are saved to `./output` with the original file names, and the script reports the GPU, ms/batch, ms/frame and FPS. Results do not depend on the batch size.

On an NVIDIA L40S (batch size 8, 480 diffusion steps): 13.5 s/batch, 1.69 s/frame, 0.59 FPS.

> [!NOTE]
> JAX does not support CUDA GPUs on native Windows. On Windows, use [WSL2](https://learn.microsoft.com/windows/wsl/install).