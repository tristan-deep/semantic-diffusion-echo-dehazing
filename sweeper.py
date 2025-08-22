"""

NOTE: pip install optuna

"""

import dataclasses
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import numpy as np
import optuna
import tyro
import yaml
import zea
from keras import ops
from PIL import Image
from zea import init_device, log

from eval import evaluate
from main import init, run


def load_images_from_dir(input_folder):
    """Load images from directory, similar to main.py implementation."""
    paths = list(Path(input_folder).glob("*.png"))

    images = []
    for path in paths:
        image = zea.io_lib.load_image(path)
        images.append(image)

    if len(images) == 0:
        raise ValueError(f"No PNG images found in {input_folder}")

    images = ops.stack(images, axis=0)
    return images, paths


def save_images_to_temp_dir(images, image_paths, prefix=""):
    """Save numpy arrays as PNG images to a temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    temp_dir_path = Path(temp_dir)

    for i, (img, path) in enumerate(zip(images, image_paths)):
        # Get the filename from the original path
        filename = Path(path).name

        # Convert image to uint8 if needed
        if img.dtype != np.uint8:
            # Assume image is in [0, 1] range and convert to [0, 255]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Ensure image is 2D or 3D
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)

        # Save as PNG
        img_pil = Image.fromarray(img)
        save_path = temp_dir_path / filename
        img_pil.save(save_path)

    return str(temp_dir_path)


@dataclasses.dataclass
class SweeperConfig:
    """Configuration for hyperparameter sweeping with Optuna."""

    # Required paths - no defaults
    input_image_dir: str  # Path to input hazy images
    roi_folder: str  # Path to ROI mask images
    reference_folder: str  # Path to reference/ground truth images
    base_config_path: str = "configs/semantic_dps.yaml"

    # Base configuration
    method: str = "semantic_dps"  # Which method to optimize
    broad_sweep: bool = False  # Choose between broad or narrow sweep

    # Optuna settings
    study_name: str = "dehaze_optimization"
    storage: Optional[str] = None  # e.g., "sqlite:///dehaze_study.db" for persistence
    n_trials: int = 100

    # Optimization settings
    objective_metric: str = "final_score"  # Which metric to optimize
    direction: str = "maximize"  # "maximize" or "minimize"

    # Output settings
    output_dir: str = "sweep_results"

    # Evaluation settings
    skip_fid: bool = False

    # Device configuration
    device: str = "auto:1"

    # Pruning settings
    enable_pruning: bool = True
    pruner_type: str = "median"  # "median", "hyperband", or "none"


class OptunaObjective:
    """Optuna objective function for hyperparameter optimization."""

    def __init__(self, config: SweeperConfig):
        self.config = config
        self.base_config = self._load_base_config()
        self.hazy_images, self.image_paths = load_images_from_dir(
            config.input_image_dir
        )

        # Initialize device
        init_device(config.device)

        # Initialize the diffusion model once
        self.diffusion_model = init(self.base_config)

    def _load_base_config(self):
        """Load base configuration from YAML file."""
        with open(self.config.base_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return zea.Config(**config_dict)

    def _create_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Create trial parameters by suggesting hyperparameters."""
        params = {
            "guidance_kwargs": {
                "omega": trial.suggest_float("omega", 0.5, 50.0, log=True),
                "omega_vent": trial.suggest_float("omega_vent", 0.0001, 50.0, log=True),
                "omega_sept": trial.suggest_float("omega_sept", 0.1, 50.0, log=True),
                "eta": trial.suggest_float("eta", 0.001, 1.0, log=True),
                "smooth_l1_beta": trial.suggest_float(
                    "smooth_l1_beta", 0.1, 10.0, log=True
                ),
            },
            "skeleton_params": {
                "sigma_pre": trial.suggest_float("skeleton_sigma_pre", 0.0, 10.0),
                "sigma_post": trial.suggest_float("skeleton_sigma_post", 0.0, 10.0),
                "threshold": trial.suggest_float("skeleton_threshold", 0.0, 1.0),
            },
            "mask_params": {
                "threshold": trial.suggest_float("mask_threshold", 0.0, 1.0),
                "sigma": trial.suggest_float("mask_sigma", 0.0, 10.0),
            },
        }

        # Add base config parameters that aren't being optimized
        if hasattr(self.base_config, "params"):
            base_params = self.base_config.params
            for key, value in base_params.items():
                if key not in params:
                    params[key] = value

        return params

    def __call__(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Suggest hyperparameters for this trial
        params = self._create_trial_params(trial)

        # Create seed for reproducibility
        seed = jax.random.PRNGKey(self.base_config.seed + trial.number)

        # Run the semantic DPS method
        try:
            hazy_images, pred_tissue_images, pred_haze_images, masks = run(
                hazy_images=self.hazy_images,
                diffusion_model=self.diffusion_model,
                seed=seed,
                **params,
            )
        except Exception as e:
            log.error(f"Error during model inference: {e}")
            return 0.0

        # Convert tensors to numpy arrays if needed
        if hasattr(pred_tissue_images, "numpy"):
            pred_tissue_images = pred_tissue_images.numpy()

        # Initialize temp directory
        pred_tissue_temp_dir = None

        try:
            # Save predicted tissue images to temp directory
            pred_tissue_temp_dir = save_images_to_temp_dir(
                pred_tissue_images, self.image_paths, prefix="pred_tissue_"
            )

            # Run evaluation using the updated evaluate function
            results = evaluate(
                folder=pred_tissue_temp_dir,
                noisy_folder=self.config.input_image_dir,
                roi_folder=self.config.roi_folder,
                reference_folder=self.config.reference_folder,
            )

            objective_value = results[self.config.objective_metric]

        except Exception as e:
            log.error(f"Error during evaluation: {e}")
            objective_value = 0.0

        finally:
            # Clean up temporary directory
            if pred_tissue_temp_dir and Path(pred_tissue_temp_dir).exists():
                try:
                    shutil.rmtree(pred_tissue_temp_dir)
                except Exception as e:
                    log.warning(
                        f"Failed to clean up temp directory {pred_tissue_temp_dir}: {e}"
                    )

        # Log intermediate results for potential pruning
        trial.report(objective_value, step=0)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Store hyperparameters as user attributes
        for key, value in params.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    trial.set_user_attr(f"{key}_{subkey}", subvalue)
            else:
                trial.set_user_attr(key, value)

        log.info(
            f"Trial {trial.number}: {self.config.objective_metric} = {objective_value:.4f}"
        )

        return objective_value


def create_pruner(pruner_type: str) -> optuna.pruners.BasePruner:
    """Create an Optuna pruner based on the specified type."""
    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=0, interval_steps=1
        )
    elif pruner_type == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=100, reduction_factor=3
        )
    else:
        return optuna.pruners.NopPruner()


def run_optimization(config: SweeperConfig):
    """Run hyperparameter optimization using Optuna."""

    # Create pruner
    pruner = create_pruner(config.pruner_type) if config.enable_pruning else None

    # Create or load study
    study = optuna.create_study(
        study_name=config.study_name,
        storage=config.storage,
        direction=config.direction,
        pruner=pruner,
        load_if_exists=True,
    )

    log.info(f"Starting optimization for method: {config.method}")
    log.info(f"Study name: {config.study_name}")
    log.info(f"Number of trials: {config.n_trials}")
    log.info(f"Objective metric: {config.objective_metric} ({config.direction})")

    # Create objective function
    objective = OptunaObjective(config)

    # Run optimization
    study.optimize(objective, n_trials=config.n_trials)

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best trial info
    best_trial = study.best_trial
    best_results = {
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs,
        "study_stats": {
            "n_trials": len(study.trials),
            "n_complete_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "n_pruned_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            ),
            "n_failed_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            ),
        },
    }

    with open(
        output_dir / f"best_results_{config.method}_{config.study_name}.json", "w"
    ) as f:
        json.dump(best_results, f, indent=2)

    # Save all trials data
    trials_data = []
    for trial in study.trials:
        trial_data = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
            "state": trial.state.name,
            "datetime_start": trial.datetime_start.isoformat()
            if trial.datetime_start
            else None,
            "datetime_complete": trial.datetime_complete.isoformat()
            if trial.datetime_complete
            else None,
        }
        trials_data.append(trial_data)

    with open(
        output_dir / f"all_trials_{config.method}_{config.study_name}.json", "w"
    ) as f:
        json.dump(trials_data, f, indent=2)

    # Print summary
    log.success("Optimization completed!")
    log.info(f"Best {config.objective_metric}: {best_trial.value:.4f}")
    log.info("Best parameters:")
    for key, value in best_trial.params.items():
        log.info(f"  {key}: {value}")

    # Print study statistics
    stats = best_results["study_stats"]
    log.info("Study statistics:")
    log.info(f"  Total trials: {stats['n_trials']}")
    log.info(f"  Complete trials: {stats['n_complete_trials']}")
    log.info(f"  Pruned trials: {stats['n_pruned_trials']}")
    log.info(f"  Failed trials: {stats['n_failed_trials']}")

    return study


def main():
    """Main function for running hyperparameter optimization."""
    config = tyro.cli(SweeperConfig)

    # Validate required paths exist
    required_paths = [
        (config.input_image_dir, "Input image directory"),
        (config.roi_folder, "ROI folder"),
        (config.reference_folder, "Reference folder"),
    ]

    for path, description in required_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"{description} not found: {path}")

    # Set visualization style
    zea.visualize.set_mpl_style()

    # Run optimization
    study = run_optimization(config)

    # Optionally, generate optimization plots
    try:
        import matplotlib.pyplot as plt
        import optuna.visualization as vis

        output_dir = Path(config.output_dir)

        # Plot optimization history
        fig = vis.matplotlib.plot_optimization_history(study).figure
        fig.savefig(
            output_dir / f"optimization_history_{config.method}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Plot parameter importances
        fig = vis.matplotlib.plot_param_importances(study).figure
        fig.savefig(
            output_dir / f"param_importances_{config.method}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Plot parallel coordinate
        fig = vis.matplotlib.plot_parallel_coordinate(study).figure
        fig.savefig(
            output_dir / f"parallel_coordinate_{config.method}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        log.success(f"Optimization plots saved to {output_dir}")

    except ImportError:
        log.warning(
            "Optuna visualization not available. Install with: pip install optuna[visualization]"
        )


if __name__ == "__main__":
    main()
