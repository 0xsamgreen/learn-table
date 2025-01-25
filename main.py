import torch
import torch.nn as nn
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate a random table (1000x10) with positive real numbers
# This simulates a large table of data we want to compress using a neural network
table = torch.rand(1000, 10) * 10  # Scale to range [0, 10] for more realistic values

# Create dataset by converting the 2D table into training samples
# Each sample will consist of:
# 1. The row and column indices (position in the table)
# 2. The actual value at that position (target)
rows = []
columns = []
targets = []

# Convert table indices to training data
# We create a sample for each cell in the table
for i in range(table.shape[0]):
    for j in range(table.shape[1]):
        rows.append(i)
        columns.append(j)
        targets.append(table[i, j])

# Convert indices to integers for embedding
# Embeddings require integer indices, as they learn a unique vector for each discrete position
rows_idx = torch.tensor(rows, dtype=torch.long)
columns_idx = torch.tensor(columns, dtype=torch.long)

# Create normalized continuous features
# Normalization is crucial for neural networks as it:
# 1. Helps prevent numerical instability
# 2. Makes the learning process more efficient
# 3. Ensures all features are on a similar scale
rows_norm = rows_idx.float() / table.shape[0]     # Normalize to [0, 1]
columns_norm = columns_idx.float() / table.shape[1]

# Stack the normalized positions into a single feature tensor
# This gives us continuous representations of the positions
X = torch.stack([rows_norm, columns_norm], dim=1)
targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

# Normalize targets (the table values)
# This is important because:
# 1. It makes the loss function's job easier
# 2. Helps prevent vanishing/exploding gradients
# 3. Makes the learning rate more consistent across different scales of data
targets_mean = targets.mean()
targets_std = targets.std()
targets_normalized = (targets - targets_mean) / targets_std  # Convert to standard normal distribution

class CompactTablePredictor(nn.Module):
    """
    Neural network that learns to predict table values from position information.
    
    Key components:
    1. Embeddings: Learn rich representations of row and column positions
    2. Continuous features: Direct normalized position information
    3. Dense layers: Combine and process all features to predict values
    
    The network uses both embeddings and continuous positions because:
    - Embeddings can capture complex patterns specific to each row/column
    - Continuous positions help generalize and interpolate between positions
    - This dual representation often leads to better predictions
    """
    def __init__(self, num_rows, num_cols, embedding_dim=4):
        super(CompactTablePredictor, self).__init__()
        
        # Embeddings learn a unique vector for each row and column
        # This allows the network to:
        # 1. Capture position-specific patterns
        # 2. Learn relationships between different positions
        # 3. Represent discrete positions in a continuous space
        self.row_embedding = nn.Embedding(num_rows, embedding_dim)
        self.col_embedding = nn.Embedding(num_cols, embedding_dim)
        
        # Neural network architecture
        # Input features: 2 normalized positions + 2 embeddings of size embedding_dim
        # The architecture is designed to be:
        # 1. Deep enough to learn complex patterns
        # 2. Narrow enough to force compression
        # 3. Regularized with LayerNorm to prevent overfitting
        self.network = nn.Sequential(
            # First layer expands the features to learn rich representations
            nn.Linear(2 + 2*embedding_dim, 32),
            nn.LayerNorm(32),  # Normalize activations for stable training
            nn.GELU(),  # Modern activation function, smoother than ReLU
            
            # Middle layer for additional pattern processing
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            
            # Final layer produces a single predicted value
            nn.Linear(16, 1)
        )
        
        # Initialize weights using modern techniques
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        Custom weight initialization for better training:
        - Kaiming initialization for linear layers (good for GELU)
        - Small normal initialization for embeddings (prevents large initial values)
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x, row_idx, col_idx):
        """
        Forward pass of the network:
        1. Get learned embeddings for the row and column positions
        2. Combine with normalized position features
        3. Process through neural network to predict value
        """
        # Get learned representations for discrete positions
        row_emb = self.row_embedding(row_idx)
        col_emb = self.col_embedding(col_idx)
        
        # Combine continuous positions with learned embeddings
        # This gives the network both explicit position information
        # and learned patterns about each position
        combined = torch.cat([x, row_emb, col_emb], dim=1)
        
        return self.network(combined)

# Initialize model, loss function, and optimizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = CompactTablePredictor(table.shape[0], table.shape[1]).to(device)

# Print model diagnostics
total_params = sum(p.numel() for p in model.parameters())
table_size = table.shape[0] * table.shape[1]
print("\nSize diagnostics:")
print(f"Number of model parameters: {total_params:,}")
print(f"Number of values in table: {table_size:,}")
print(f"Ratio of parameters to table values: {total_params/table_size:.2f}\n")

# Print detailed parameter counts
embedding_params = sum(p.numel() for name, p in model.named_parameters() if 'embedding' in name)
network_params = sum(p.numel() for name, p in model.named_parameters() if 'network' in name)
print("Parameter breakdown:")
print(f"Embedding layers: {embedding_params:,}")
print(f"Network layers: {network_params:,}\n")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Move data to device
X = X.to(device)
rows_idx = rows_idx.to(device)
columns_idx = columns_idx.to(device)
targets_normalized = targets_normalized.to(device)

# Training loop
n_epochs = 300
batch_size = 128
n_batches = len(X) // batch_size

print(f"Training on device: {device}")
print("Starting training...")

for epoch in range(n_epochs):
    # Set model to training mode
    model.train()
    
    # Shuffle the data
    indices = torch.randperm(len(X))
    total_loss = 0
    
    for i in range(n_batches):
        # Get batch
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        X_batch = X[batch_indices]
        rows_batch = rows_idx[batch_indices]
        columns_batch = columns_idx[batch_indices]
        y_batch = targets_normalized[batch_indices]
        
        # Forward pass
        predictions = model(X_batch, rows_batch, columns_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / n_batches
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}")

print("Training completed!")

# Validation
print("\nValidating model predictions...")
model.eval()
with torch.no_grad():
    # Test 10 positions including corners, middle, and random spots
    test_positions = [
        (0, 0),      # Top-left corner
        (0, 9),      # Top-right corner
        (999, 0),    # Bottom-left corner
        (999, 9),    # Bottom-right corner
        (500, 5),    # Middle
        (250, 3),    # Quarter way
        (750, 7),    # Three-quarters way
        (123, 4),    # Random position 1
        (867, 2),    # Random position 2
        (432, 8),    # Random position 3
    ]
    
    # Print header
    print("\n{:<15} {:<12} {:<12} {:<12}".format(
        "Position", "Predicted", "Actual", "Abs Error"))
    print("-" * 51)
    
    for row, column in test_positions:
        # Create input tensor
        x = torch.tensor([[row / table.shape[0], column / table.shape[1]]], dtype=torch.float32).to(device)
        row_idx = torch.tensor([row], dtype=torch.long).to(device)
        column_idx = torch.tensor([column], dtype=torch.long).to(device)
        
        # Get prediction and denormalize
        pred = model(x, row_idx, column_idx).item() * targets_std.item() + targets_mean.item()
        actual = table[row, column].item()
        abs_error = abs(pred - actual)
        
        # Print row
        print("{:<15} {:<12.4f} {:<12.4f} {:<12.4f}".format(
            f"({row}, {column})", pred, actual, abs_error))
