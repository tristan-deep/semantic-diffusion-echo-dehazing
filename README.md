<div align="center">
	<h1>Semantic Diffusion Posterior Sampling for Cardiac Ultrasound Dehazing</h1>
	<p>
		<a href="https://arxiv.org/abs/0000.0000"><img src="https://img.shields.io/badge/arXiv-0000.0000-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
		<a href="https://huggingface.co/collections/tristan-deep/semantic-diffusion-posterior-sampling-for-cardiac-ultrasound-68a70559a7f719c7e6bd5788"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-orange" alt="Hugging Face Model"></a>
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
	<img src="./paper/animation.gif" alt="Cardiac Ultrasound Dehazing Animation" style="max-width: 100%; height: auto;">
</p>

### Installation

The algorithm is implemented using Keras with JAX backend. Furthermore it heavily relies on the [zea ultrasound library](https://github.com/tue-bmd/zea).

Either install the following in your Python environment, or use the [Dockerfile](./Dockerfile) provided in this repository.

```bash
# requires Python>=3.10
pip install tyro optuna zea==0.0.4
pip install -U "jax[cuda12]"
```

> [!NOTE]
> Although the code was primarily tested with JAX as the Keras backend, TensorFlow and PyTorch should also work.

### Running the algorithm

Some example images are downloaded in the [./assets](./assets) folder. The models are automatically downloaded from the [Hugging Face Model Hub](https://huggingface.co/collections/tristan-deep/semantic-diffusion-posterior-sampling-for-cardiac-ultrasound-68a70559a7f719c7e6bd5788).

```bash
python main.py --input-folder ./assets --output-folder ./temp
```

Alternatively, you can use the Gradio app provided in this repository to interact with the model via a web interface. To launch the app, run:

```bash
python app.py
```
