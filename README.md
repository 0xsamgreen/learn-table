# Table Predictor

A PyTorch-based neural network implementation for learning and predicting values in a table structure. This model uses embeddings and a compact neural network architecture to efficiently learn patterns in tabular data.

## Features

- Efficient table value prediction using neural networks
- Compact architecture with embeddings for rows and columns
- Support for large tables (tested with 1000x10 dimensions)
- Normalized inputs and outputs for stable training
- GPU support (MPS/CUDA) when available

## Requirements

- Python 3.6+
- PyTorch 2.0.0+
- NumPy 1.21.0+

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main implementation is in `main.py`. The model:

- Creates a random table for training
- Uses embeddings to represent row and column positions
- Trains a neural network to predict table values
- Includes validation to test predictions

The `CompactTablePredictor` class provides the core functionality:
- `__init__`: Initialize model with number of rows, columns, and embedding dimension
- `forward`: Make predictions given input coordinates
- Built-in weight initialization for optimal training

## Model Architecture

- Row and column embeddings
- Concatenated with normalized position features
- Three-layer neural network with LayerNorm and GELU activations
- MSE loss function with AdamW optimizer
