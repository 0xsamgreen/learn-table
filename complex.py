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
    """
    def __init__(self, num_rows, num_cols, embedding_dim=4):
        super(CompactTablePredictor, self).__init__()
        
        # Embeddings learn a unique vector for each row and column
        self.row_embedding = nn.Embedding(num_rows, embedding_dim)
        self.col_embedding = nn.Embedding(num_cols, embedding_dim)
        
        # Neural network architecture
        self.network = nn.Sequential(
            nn.Linear(2 + 2*embedding_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            
            nn.Linear(16, 1)
        )
        
        # Initialize weights using modern techniques
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        Custom weight initialization for better training
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x, row_idx, col_idx):
        """Forward pass of the network"""
        row_emb = self.row_embedding(row_idx)
        col_emb = self.col_embedding(col_idx)
        combined = torch.cat([x, row_emb, col_emb], dim=1)
        return self.network(combined)

class ExperienceBuffer:
    """Buffer to store past training examples to prevent catastrophic forgetting"""
    def __init__(self, max_size=5000):
        self.max_size = max_size
        self.buffer = []
        
    def add_experience(self, indices, x, rows, cols, targets):
        """Add new experiences to the buffer"""
        for i in range(len(indices)):
            experience = {
                'index': indices[i].item(),
                'x': x[i].clone(),
                'row': rows[i].clone(),
                'col': cols[i].clone(),
                'target': targets[i].clone()
            }
            if len(self.buffer) < self.max_size:
                self.buffer.append(experience)
            else:
                # Replace random old experience
                idx = torch.randint(0, len(self.buffer), (1,)).item()
                self.buffer[idx] = experience
                
    def sample_experience(self, batch_size):
        """Sample random experiences from buffer"""
        if len(self.buffer) == 0:
            return None
        
        indices = torch.randint(0, len(self.buffer), (min(batch_size, len(self.buffer)),))
        experiences = [self.buffer[i] for i in indices]
        
        # Collate experiences into batches
        x = torch.stack([e['x'] for e in experiences])
        rows = torch.tensor([e['row'] for e in experiences], dtype=torch.long)
        cols = torch.tensor([e['col'] for e in experiences], dtype=torch.long)
        targets = torch.stack([e['target'] for e in experiences])
        
        return x, rows, cols, targets

def select_training_samples(total_samples, percentage=0.1):
    """Select a random subset of indices for training"""
    num_samples = int(total_samples * percentage)
    indices = torch.randperm(total_samples)[:num_samples]
    return indices

def validate_random_samples(model, table, device, targets_mean, targets_std, n_samples=10):
    """Validate model on random samples and print results"""
    model.eval()
    with torch.no_grad():
        # Generate random positions
        rows = torch.randint(0, table.shape[0], (n_samples,))
        cols = torch.randint(0, table.shape[1], (n_samples,))
        
        print("\nRandom Validation Results:")
        print("{:<15} {:<12} {:<12} {:<12}".format(
            "Position", "Predicted", "Actual", "Abs Error"))
        print("-" * 51)
        
        for i in range(n_samples):
            row, col = rows[i].item(), cols[i].item()
            
            # Prepare input
            x = torch.tensor([[row / table.shape[0], col / table.shape[1]]], 
                           dtype=torch.float32).to(device)
            row_idx = torch.tensor([row], dtype=torch.long).to(device)
            col_idx = torch.tensor([col], dtype=torch.long).to(device)
            
            # Get prediction and denormalize
            pred = model(x, row_idx, col_idx).item() * targets_std.item() + targets_mean.item()
            actual = table[row, col].item()
            abs_error = abs(pred - actual)
            
            print("{:<15} {:<12.4f} {:<12.4f} {:<12.4f}".format(
                f"({row}, {col})", pred, actual, abs_error))
    
    model.train()

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

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Move data to device
X = X.to(device)
rows_idx = rows_idx.to(device)
columns_idx = columns_idx.to(device)
targets_normalized = targets_normalized.to(device)

# Initialize experience buffer
experience_buffer = ExperienceBuffer()

# Training loop
n_epochs = 1000
batch_size = 128
replay_ratio = 0.3  # Portion of batch that comes from replay buffer
print(f"Training on device: {device}")
print("Starting training...")

for epoch in range(n_epochs):
    # Set model to training mode
    model.train()
    
    # Select 10% of data for this epoch
    train_indices = select_training_samples(len(X))
    
    # Get training data for this epoch
    X_train = X[train_indices]
    rows_train = rows_idx[train_indices]
    cols_train = columns_idx[train_indices]
    targets_train = targets_normalized[train_indices]
    
    # Add current samples to experience buffer
    experience_buffer.add_experience(
        train_indices, X_train, rows_train, cols_train, targets_train
    )
    
    # Shuffle the selected data
    shuffle_idx = torch.randperm(len(X_train))
    X_train = X_train[shuffle_idx]
    rows_train = rows_train[shuffle_idx]
    cols_train = cols_train[shuffle_idx]
    targets_train = targets_train[shuffle_idx]
    
    # Calculate number of batches for the selected data
    n_batches = len(X_train) // batch_size
    total_loss = 0
    
    for i in range(n_batches):
        # Calculate sizes for current and replay samples
        replay_size = int(batch_size * replay_ratio)
        current_size = batch_size - replay_size
        
        # Get current batch
        start_idx = i * current_size
        end_idx = start_idx + current_size
        X_batch = X_train[start_idx:end_idx]
        rows_batch = rows_train[start_idx:end_idx]
        cols_batch = cols_train[start_idx:end_idx]
        y_batch = targets_train[start_idx:end_idx]
        
        # Get replay batch and combine with current batch
        replay_data = experience_buffer.sample_experience(replay_size)
        if replay_data is not None:
            X_replay, rows_replay, cols_replay, targets_replay = replay_data
            X_batch = torch.cat([X_batch, X_replay.to(device)])
            rows_batch = torch.cat([rows_batch, rows_replay.to(device)])
            cols_batch = torch.cat([cols_batch, cols_replay.to(device)])
            y_batch = torch.cat([y_batch, targets_replay.to(device)])
        
        # Forward pass
        predictions = model(X_batch, rows_batch, cols_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / n_batches
    
    # Print progress and validate every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"\nEpoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}")
        validate_random_samples(model, table, device, targets_mean, targets_std)

print("\nTraining completed!")

# Final validation
print("\nFinal validation results:")
validate_random_samples(model, table, device, targets_mean, targets_std, n_samples=10)
