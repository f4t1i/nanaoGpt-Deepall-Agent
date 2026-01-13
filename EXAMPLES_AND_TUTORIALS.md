# ARS Optimizer: Code Examples & Tutorials

**Document Version:** 1.0  
**Date:** January 13, 2026  
**Status:** ✓ PRODUCTION READY  
**Target Audience:** Developers, ML Engineers, Data Scientists

---

## 📋 Table of Contents

1. [Basic Training Example](#basic-training-example)
2. [Advanced Monitoring Tutorial](#advanced-monitoring-tutorial)
3. [Multi-GPU Training Example](#multi-gpu-training-example)
4. [Hyperparameter Tuning Tutorial](#hyperparameter-tuning-tutorial)
5. [Custom Loss Functions](#custom-loss-functions)
6. [Integration with Popular Frameworks](#integration-with-popular-frameworks)
7. [Production Patterns](#production-patterns)

---

## 🎯 Basic Training Example

This example demonstrates the simplest way to use ARS Optimizer with a basic neural network.

### Complete Training Script

```python
# basic_training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ars_optimizer import ARSOptimizer

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Create synthetic dataset
print("Creating dataset...")
X_train = torch.randn(1000, 20)
y_train = torch.randn(1000, 1)
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Step 2: Define model
print("Building model...")
class SimpleModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleModel().to(DEVICE)

# Step 3: Create optimizer (ARS Optimizer)
print("Initializing ARS Optimizer...")
optimizer = ARSOptimizer(
    model.parameters(),
    lr=LEARNING_RATE,
    entropy_threshold=0.7,
    surprise_scale=0.01,
    jitter_scale=0.01
)

# Step 4: Define loss function
criterion = nn.MSELoss()

# Step 5: Training loop
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        # Move data to device
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

print("✓ Training completed successfully!")

# Step 6: Save model
torch.save(model.state_dict(), 'model.pth')
print("✓ Model saved to model.pth")
```

### Running the Example

```bash
# Install dependencies
pip install torch

# Run training
python basic_training.py
```

---

## 📊 Advanced Monitoring Tutorial

This tutorial shows how to monitor ARS Optimizer's internal mechanisms during training.

### Monitoring Script

```python
# advanced_monitoring.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ars_optimizer import ARSOptimizer
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Create dataset
X_train = torch.randn(1000, 20)
y_train = torch.randn(1000, 1)
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create model
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Create optimizer
optimizer = ARSOptimizer(
    model.parameters(),
    lr=LEARNING_RATE,
    entropy_threshold=0.7,
    surprise_scale=0.01,
    jitter_scale=0.01
)

criterion = nn.MSELoss()

# Monitoring class
class TrainingMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def log_step(self, **kwargs):
        """Log metrics for each training step"""
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def log_epoch(self, epoch, **kwargs):
        """Log metrics for each epoch"""
        print(f"\nEpoch {epoch + 1}:")
        for key, value in kwargs.items():
            print(f"  {key}: {value:.6f}")
    
    def plot_metrics(self, save_path='training_metrics.png'):
        """Plot all collected metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Loss
        if 'loss' in self.metrics:
            axes[0, 0].plot(self.metrics['loss'], linewidth=1.5)
            axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Entropy Guard (Ψ_t)
        if 'entropy_guard' in self.metrics:
            axes[0, 1].plot(self.metrics['entropy_guard'], linewidth=1.5, color='orange')
            axes[0, 1].set_title('Entropy Guard (Ψ_t)', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].axhline(y=0.7, color='r', linestyle='--', label='Threshold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Surprise Gate (Φ_t)
        if 'surprise_gate' in self.metrics:
            axes[1, 0].plot(self.metrics['surprise_gate'], linewidth=1.5, color='green')
            axes[1, 0].set_title('Surprise Gate (Φ_t)', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Chronos-Jitter (χ_t)
        if 'chronos_jitter' in self.metrics:
            axes[1, 1].plot(self.metrics['chronos_jitter'], linewidth=1.5, color='red')
            axes[1, 1].set_title('Chronos-Jitter (χ_t)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✓ Metrics saved to {save_path}")

# Initialize monitor
monitor = TrainingMonitor()

# Training loop with monitoring
print("Starting training with monitoring...")
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log metrics
        monitor.log_step(
            loss=loss.item(),
            entropy_guard=optimizer.entropy_guard,
            surprise_gate=optimizer.surprise_gate,
            chronos_jitter=optimizer.chronos_jitter
        )
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    if (epoch + 1) % 10 == 0:
        monitor.log_epoch(
            epoch,
            avg_loss=avg_loss,
            entropy_guard=optimizer.entropy_guard,
            surprise_gate=optimizer.surprise_gate
        )

# Plot results
monitor.plot_metrics()
print("✓ Training completed!")
```

---

## 🚀 Multi-GPU Training Example

This example demonstrates how to use ARS Optimizer with multiple GPUs.

### Multi-GPU Training Script

```python
# multi_gpu_training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ars_optimizer import ARSOptimizer

# Configuration
BATCH_SIZE = 64  # Per GPU
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Check GPU availability
print(f"Available GPUs: {torch.cuda.device_count()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create dataset
X_train = torch.randn(10000, 20)
y_train = torch.randn(10000, 1)
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create model
model = nn.Sequential(
    nn.Linear(20, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

# Wrap model for multi-GPU if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

# Create optimizer
optimizer = ARSOptimizer(
    model.parameters(),
    lr=LEARNING_RATE,
    entropy_threshold=0.7,
    surprise_scale=0.01,
    jitter_scale=0.01
)

criterion = nn.MSELoss()

# Training loop
print("Starting multi-GPU training...")
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

print("✓ Multi-GPU training completed!")
```

---

## 🔧 Hyperparameter Tuning Tutorial

This tutorial demonstrates how to systematically tune ARS Optimizer hyperparameters.

### Hyperparameter Tuning Script

```python
# hyperparameter_tuning.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ars_optimizer import ARSOptimizer
import itertools
import json

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 30

# Create dataset
X_train = torch.randn(1000, 20)
y_train = torch.randn(1000, 1)
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Hyperparameter ranges to test
param_grid = {
    'lr': [0.0001, 0.001, 0.01],
    'entropy_threshold': [0.6, 0.7, 0.8],
    'surprise_scale': [0.005, 0.01, 0.02],
    'jitter_scale': [0.005, 0.01, 0.02],
}

def train_with_params(params):
    """Train model with given hyperparameters"""
    # Create fresh model
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    # Create optimizer with given parameters
    optimizer = ARSOptimizer(model.parameters(), **params)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    final_loss = total_loss / len(dataloader)
    return final_loss

# Grid search
print("Starting hyperparameter grid search...")
results = []

param_combinations = list(itertools.product(*param_grid.values()))
total_combinations = len(param_combinations)

for idx, values in enumerate(param_combinations):
    params = dict(zip(param_grid.keys(), values))
    
    print(f"\n[{idx + 1}/{total_combinations}] Testing parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    final_loss = train_with_params(params)
    print(f"  Final Loss: {final_loss:.6f}")
    
    results.append({
        'params': params,
        'final_loss': final_loss
    })

# Find best parameters
best_result = min(results, key=lambda x: x['final_loss'])
print("\n" + "="*50)
print("BEST PARAMETERS:")
for key, value in best_result['params'].items():
    print(f"  {key}: {value}")
print(f"Final Loss: {best_result['final_loss']:.6f}")
print("="*50)

# Save results
with open('hyperparameter_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Results saved to hyperparameter_results.json")
```

---

## 🎨 Custom Loss Functions

This section shows how to use ARS Optimizer with custom loss functions.

### Example: Custom Loss Function

```python
# custom_loss_functions.py
import torch
import torch.nn as nn
from ars_optimizer import ARSOptimizer

# Custom loss function: Huber Loss
class CustomHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, predictions, targets):
        diff = torch.abs(predictions - targets)
        mask = diff <= self.delta
        
        # Quadratic part
        quad_loss = 0.5 * diff ** 2
        
        # Linear part
        linear_loss = self.delta * (diff - 0.5 * self.delta)
        
        loss = torch.where(mask, quad_loss, linear_loss)
        return loss.mean()

# Create model and optimizer
model = nn.Linear(10, 1)
optimizer = ARSOptimizer(model.parameters(), lr=0.001)
criterion = CustomHuberLoss(delta=1.0)

# Create dummy data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

print("✓ Training with custom loss function completed!")
```

---

## 🔗 Integration with Popular Frameworks

### PyTorch Lightning Integration

```python
# lightning_integration.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ars_optimizer import ARSOptimizer

class ARSLightningModule(pl.LightningModule):
    def __init__(self, input_size=20, hidden_size=64, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        predictions = self(X)
        loss = self.criterion(predictions, y)
        return loss
    
    def configure_optimizers(self):
        return ARSOptimizer(
            self.parameters(),
            lr=self.learning_rate,
            entropy_threshold=0.7,
            surprise_scale=0.01,
            jitter_scale=0.01
        )

# Create dataset
X_train = torch.randn(1000, 20)
y_train = torch.randn(1000, 1)
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32)

# Create model and trainer
model = ARSLightningModule()
trainer = pl.Trainer(max_epochs=50, accelerator='gpu' if torch.cuda.is_available() else 'cpu')

# Train
trainer.fit(model, dataloader)
print("✓ PyTorch Lightning training completed!")
```

---

## 🏭 Production Patterns

### Pattern 1: Checkpoint Management

```python
# checkpoint_management.py
import torch
from ars_optimizer import ARSOptimizer

def save_checkpoint(model, optimizer, epoch, loss, path='checkpoint.pt'):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path='checkpoint.pt'):
    """Load training checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['epoch'], checkpoint['loss']

# Usage
model = torch.nn.Linear(10, 1)
optimizer = ARSOptimizer(model.parameters())

# Save checkpoint
save_checkpoint(model, optimizer, epoch=10, loss=0.05)

# Load checkpoint
epoch, loss = load_checkpoint(model, optimizer)
print(f"Resumed from epoch {epoch}, loss: {loss:.4f}")
```

### Pattern 2: Early Stopping

```python
# early_stopping.py
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

# Usage in training loop
early_stopping = EarlyStopping(patience=5)

for epoch in range(num_epochs):
    # Training code...
    val_loss = validate(model, val_dataloader)
    
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 📚 Additional Resources

- **Official Documentation:** `ARS_TECHNICAL_ARCHITECTURE.md`
- **Deployment Guide:** `DEPLOYMENT_GUIDE.md`
- **Implementation Roadmap:** `ARS_IMPLEMENTATION_ROADMAP.md`

---

**Document Version:** 1.0  
**Status:** ✓ PRODUCTION READY  
**Last Updated:** January 13, 2026  
**Author:** Manus AI
