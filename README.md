# Code Structure
CST-VT-ENSO-Forecast/
│
├── README.md                # Project overview, installation, and usage instructions
│
├── data/
│   └── causal_graph.csv     # Placeholder for the causal adjacency matrix
│
├── src/
│   ├── __init__.py
│   ├── model.py             # Contains the core CST-VT model architecture
│   ├── dataset.py           # PyTorch dataset and dataloader logic
│   ├── trainer.py           # The main training and evaluation loop
│   ├── utils.py             # Utility functions (e.g., metrics, logging)
│   └── config.py            # Central configuration file for hyperparameters
│
└── run_experiment.py        # Main script to start a training/evaluation run

# A Causal-Informed Spatio-Temporal Variational Transformer for Probabilistic Multi-Year ENSO Forecasting

This repository contains the official PyTorch implementation for the paper: **"A Causal-Informed Spatio-Temporal Variational Transformer for Probabilistic Multi-Year ENSO Forecasting"**.

Our proposed model, the **CST-VT**, is a novel deep learning framework designed to push the frontiers of El Niño-Southern Oscillation (ENSO) predictability. It leverages a Transformer architecture to capture long-range spatio-temporal dependencies, incorporates a physics-guided causal graph to enhance interpretability and performance, and provides probabilistic forecasts to quantify uncertainty.

## Model Architecture

The CST-VT consists of four main components:
1.  **Spatio-Temporal Embedding Layer**: Inspired by Vision Transformers (ViT), input global climate maps are partitioned into patches and embedded into a high-dimensional space.
2.  **Causal-Informed Transformer Encoder**: A stack of Transformer layers where the self-attention mechanism is constrained by a pre-defined causal adjacency graph, guiding the model to learn physically meaningful teleconnections.
3.  **Variational Probabilistic Head**: A decoder that outputs the parameters (mean and variance) of a Gaussian distribution for the target Niño3.4 index, enabling probabilistic forecasting.
4.  **Two-Stage Training**: The model is first pre-trained on a large volume of CMIP6 climate model data and then fine-tuned on historical reanalysis data to adapt to real-world observations.

## Requirements

The codebase is built using Python 3.8+ and PyTorch 1.10+. The main dependencies are listed below. You can install them using pip:

```bash
pip install torch torchvision numpy pandas xarray tqdm pyyaml

# Dataset Preparation

/path/to/your/data/
├── cmip6/
│   ├── sst/
│   │   ├── model1.nc
│   │   └── ...
│   ├── ohc/
│   └── ...
├── soda/
│   └── ...
└── godas/
    └── ...
