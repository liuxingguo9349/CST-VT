# A Causal-Informed Spatio-Temporal Variational Transformer for Probabilistic Multi-Year ENSO Forecasting

This repository contains the official PyTorch implementation for the paper: **"A Causal-Informed Spatio-Temporal Variational Transformer for Probabilistic Multi-Year ENSO Forecasting"**.

Our proposed model, the **CST-VT**, is a novel deep learning framework designed to push the frontiers of El Niño-Southern Oscillation (ENSO) predictability. It leverages a Transformer architecture to capture long-range spatio-temporal dependencies, incorporates a physics-guided causal graph to enhance interpretability and performance, and provides probabilistic forecasts to quantify uncertainty.

## Features
- **Transformer-based Backbone**: Utilizes the power of self-attention to model global-scale, long-range climate teleconnections.
- **Causal-Informed Attention**: Integrates a physics-based causal graph as an inductive bias, improving performance and model interpretability.
- **Probabilistic Forecasting**: Employs a variational inference framework to produce not just point forecasts, but full predictive distributions (mean and variance).
- **Two-Stage Transfer Learning**: Pre-trained on a large corpus of CMIP6 climate model data, then fine-tuned on historical reanalysis data for robust adaptation.

## Getting Started

This guide will walk you through setting up the environment, preparing the data, and running experiments.

### 1. Prerequisites

The codebase is built using **Python 3.8+** and **PyTorch 1.10+**. Ensure you have a compatible environment. We recommend using a virtual environment (e.g., `conda` or `venv`).

### 2. Installation

Clone this repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/CST-VT-ENSO-Forecast.git
cd CST-VT-ENSO-Forecast

# Install dependencies
pip install -r requirements.txt
```
The `requirements.txt` file should contain:
```
torch>=1.10.0
torchvision
numpy
pandas
xarray
tqdm
pyyaml
einops
```
For GPU support, please ensure you have a compatible version of CUDA installed and follow the official PyTorch instructions to install the appropriate version.

### 3. Project Structure

The project is organized in a modular structure to promote clarity and ease of use:

```
CST-VT-ENSO-Forecast/
│
├── README.md                # You are here
├── requirements.txt         # Python dependencies
│
├── data/
│   └── causal_graph.csv     # The causal adjacency matrix
│
├── src/
│   ├── model.py             # Core CST-VT model architecture
│   ├── dataset.py           # PyTorch Dataset and DataLoader logic
│   ├── trainer.py           # Training, validation, and evaluation loops
│   ├── utils.py             # Metrics, optimizers, and helper functions
│   └── config.py            # Central configuration for all parameters
│
└── run_experiment.py        # Main executable script
```

### 4. Data Preparation

Due to their large size, the climate datasets (CMIP6, SODA, GODAS, ERA5) are not included in this repository. You must download them from their respective official sources.

**Step 1: Download Data**
- **CMIP6**: Earth System Grid Federation (ESGF) portal.
- **SODA, GODAS, ERA5**: Official provider websites.

**Step 2: Preprocess and Structure Data**
All downloaded data must be preprocessed into a consistent format. This includes:
- Calculating monthly anomalies.
- Interpolating all variables to a common grid (e.g., 5°x5°).
- Saving the final arrays as NetCDF (`.nc`) files.

Organize the preprocessed files into the following directory structure:
```
/path/to/your/data/
├── cmip6/                 # For pre-training
│   ├── sst_1850-2014.nc
│   ├── ohc_1850-2014.nc
│   └── ...
├── soda/                  # For fine-tuning
│   └── soda_1871-1980.nc
└── godas/                 # For validation
    └── godas_1981-2022.nc
```

**Step 3: Update Configuration**
Open `src/config.py` and update the `DATA_PATHS` dictionary to point to your dataset locations.

```python
# In src/config.py
DATA_PATHS = {
    "cmip6_pretrain": "/path/to/your/data/cmip6/",
    "soda_finetune": "/path/to/your/data/soda/",
    "godas_validate": "/path/to/your/data/godas/",
    # ...
}
```

## Running Experiments

All experiments are controlled via the `run_experiment.py` script and configured in `src/config.py`.

### 1. Configure the Experiment

Before running, you can modify hyperparameters and settings in `src/config.py`. This central file allows for easy management and reproduction of experiments. Key settings include:
- `DEVICE`: Set to `"cuda"` for GPU or `"cpu"`.
- `MODEL_PARAMS`: CST-VT architecture details (e.g., `embed_dim`, `num_layers`).
- `TRAINING_PARAMS`: Epochs, batch size, learning rates for both pre-training and fine-tuning stages.

### 2. Train a New Model

To train a new model from scratch, run the following command. This will execute the full two-stage training pipeline: pre-training on CMIP6 followed by fine-tuning on SODA.

The `--lead_time` argument specifies the forecast target in months.

```bash
# Example: Train a model to forecast 18 months ahead
python run_experiment.py --lead_time 18 --mode train
```

The script will:
1.  Load the configuration for an 18-month lead forecast.
2.  Initialize the model, optimizer, and datasets.
3.  Execute the pre-training loop.
4.  Execute the fine-tuning loop, validating against the GODAS set after each epoch.
5.  The best performing model (based on validation ACC) will be saved automatically to the `checkpoints/` directory with a name like `cst_vt_lead18_best.pth`.

### 3. Evaluate a Trained Model

Once a model is trained, you can evaluate its performance on the validation set using the `--mode evaluate` flag. You must provide the path to the saved model checkpoint.

```bash
# Example: Evaluate the best 18-month model
python run_experiment.py --lead_time 18 --mode evaluate --checkpoint_path checkpoints/cst_vt_lead18_best.pth
```

This will load the specified model, run inference on the validation dataset, and print the final performance metrics: **ACC**, **RMSE**, and **CRPS**.
