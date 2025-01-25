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

Example output:
```
Validating model predictions...

Position        Predicted    Actual       Abs Error   
---------------------------------------------------
(0, 0)          8.5921       8.8227       0.2306      
(0, 9)          3.2682       1.3319       1.9364      
(999, 0)        8.4575       7.3372       1.1203      
(999, 9)        4.4647       4.1109       0.3538      
(500, 5)        5.4440       4.8687       0.5753      
(250, 3)        3.3347       3.1535       0.1812      
(750, 7)        3.9537       4.0664       0.1127      
(123, 4)        8.1307       8.8068       0.6761      
(867, 2)        4.0347       4.6205       0.5858      
(432, 8)        6.8689       6.4951       0.3737
```

The `CompactTablePredictor` class provides the core functionality:
- `__init__`: Initialize model with number of rows, columns, and embedding dimension
- `forward`: Make predictions given input coordinates
- Built-in weight initialization for optimal training

## Model Architecture

- Row and column embeddings
- Concatenated with normalized position features
- Three-layer neural network with LayerNorm and GELU activations
- MSE loss function with AdamW optimizer
