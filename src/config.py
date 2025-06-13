# src/config.py

import yaml

# --- Project-Level Settings ---
PROJECT_NAME = "CST-VT-ENSO-Forecast"
DEVICE = "cuda"  # "cuda" or "cpu"
LOG_DIR = "logs/"
CHECKPOINT_DIR = "checkpoints/"

# --- Data Paths (MUST BE MODIFIED BY THE USER) ---
# These paths should point to the directories containing preprocessed NetCDF files.
DATA_PATHS = {
    "cmip6_pretrain": "/path/to/your/data/cmip6/",
    "soda_finetune": "/path/to/your/data/soda/",
    "godas_validate": "/path/to/your/data/godas/",
    "era5_validate": "/path/to/your/data/era5/",
    "causal_graph": "data/causal_graph.csv"
}

# --- Data Parameters ---
INPUT_VARIABLES = ['sst', 'ohc', 'u10', 'v10']
NUM_INPUT_VARIABLES = len(INPUT_VARIABLES)
INPUT_TIME_STEPS = 12  # Use 12 months of history as input
GRID_SHAPE = (36, 72)  # Target grid resolution (e.g., 5x5 degrees)

# --- Model Hyperparameters ---
MODEL_PARAMS = {
    "patch_size": 4,
    "embed_dim": 768,
    "num_layers": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "dropout_rate": 0.1,
}

# --- Training Hyperparameters ---
TRAINING_PARAMS = {
    # Pre-training on CMIP6
    "pretrain": {
        "num_epochs": 100,
        "batch_size": 64,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 0.05,
        "scheduler": "CosineAnnealingLR",
        "beta_elbo": 0.1,  # Weight for the KL divergence term in ELBO loss
    },
    # Fine-tuning on SODA
    "finetune": {
        "num_epochs": 50,
        "batch_size": 32,
        "optimizer": "AdamW",
        "learning_rate": 1e-5,
        "weight_decay": 0.05,
        "scheduler": "CosineAnnealingLR",
        "beta_elbo": 0.1,
    }
}

def get_config():
    """Returns a dictionary of all configurations."""
    config = {
        "project_name": PROJECT_NAME,
        "device": DEVICE,
        "log_dir": LOG_DIR,
        "checkpoint_dir": CHECKPOINT_DIR,
        "data_paths": DATA_PATHS,
        "data_params": {
            "variables": INPUT_VARIABLES,
            "num_variables": NUM_INPUT_VARIABLES,
            "time_steps": INPUT_TIME_STEPS,
            "grid_shape": GRID_SHAPE,
        },
        "model_params": MODEL_PARAMS,
        "training_params": TRAINING_PARAMS
    }
    return config

if __name__ == '__main__':
    # Example of how to use the config
    config = get_config()
    print(yaml.dump(config, default_flow_style=False))
