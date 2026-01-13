"""
Progressive Model Inheritance Training Loop
Combines ARS Optimizer + Fisher Information Regularization + Scaling Laws

Trains miniseries d10 → d11 → d12 → ... → d20 with:
- Progressive weight inheritance from previous model
- Fisher Information regularization to prevent forgetting
- ARS Optimizer for stable training
- Checkpoint management and metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

from regularization import (
    FisherInformationMatrix,
    ProgressiveInheritanceRegularization,
    AdaptiveRegularizationStrength,
    CombinedRegularizationLoss,
    save_model_weights,
    load_anchor_weights,
    compute_weight_change_statistics
)
from ars_optimizer import ARSOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MiniseriesModel(nn.Module):
    """
    Simple transformer-based model for miniseries training.
    Sizes: d10 (7M) → d20 (7.2B) parameters.
    """
    
    def __init__(self, vocab_size: int = 50257, hidden_size: int = 768, num_layers: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=num_layers
        )
        self.head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.head(x)
        return logits


class MiniseriesTrainer:
    """
    Trainer for progressive model inheritance with ARS and regularization.
    """
    
    def __init__(
        self,
        model_config: Dict,
        training_config: Dict,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            model_config: Model configuration (hidden_size, num_layers, etc.)
            training_config: Training configuration (lr, epochs, batch_size, etc.)
            device: Device to train on
        """
        self.model_config = model_config
        self.training_config = training_config
        self.device = device
        
        self.model = None
        self.optimizer = None
        self.ars = None
        self.fisher = None
        self.reg_loss_fn = None
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'ars_damping': [],
            'weight_changes': []
        }
        
        logger.info(f"Initialized MiniseriesTrainer on {device}")
    
    def create_model(self, hidden_size: int, num_layers: int) -> nn.Module:
        """Create model with specified architecture."""
        model = MiniseriesModel(
            vocab_size=self.model_config.get('vocab_size', 50257),
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        return model.to(self.device)
    
    def initialize_training(
        self,
        model: nn.Module,
        previous_weights: Optional[Dict] = None
    ) -> None:
        """
        Initialize optimizer and regularization for training.
        
        Args:
            model: Model to train
            previous_weights: Weights from previous model (for regularization)
        """
        self.model = model
        
        # Initialize ARS Optimizer
        self.ars = ARSOptimizer(
            entropy_window=self.training_config.get('ars_entropy_window', 10),
            surprise_window=self.training_config.get('ars_surprise_window', 10)
        )
        
        # Initialize optimizer (Adam)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.training_config.get('learning_rate', 1e-5),
            weight_decay=self.training_config.get('weight_decay', 0.0)
        )
        
        # Initialize regularization if previous weights exist
        if previous_weights is not None:
            self.fisher = FisherInformationMatrix(model, self.device)
            # Load or compute Fisher Information
            fisher_path = self.training_config.get('fisher_path')
            if fisher_path and Path(fisher_path).exists():
                self.fisher.load(fisher_path)
            
            reg_loss = ProgressiveInheritanceRegularization(
                model=model,
                fisher_information=self.fisher,
                anchor_weights=previous_weights,
                gamma=self.training_config.get('regularization_gamma', 0.1),
                device=self.device
            )
            
            self.reg_loss_fn = reg_loss
        
        logger.info("Training initialized")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch
            total_epochs: Total epochs
            
        Returns:
            Dictionary with metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Adaptive regularization strength
        adaptive_gamma = AdaptiveRegularizationStrength(
            gamma_0=self.training_config.get('regularization_gamma', 0.1)
        )
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            task_loss = nn.functional.cross_entropy(outputs, targets)
            
            # Add regularization loss if available
            total_loss_val = task_loss
            if self.reg_loss_fn is not None:
                reg_loss = self.reg_loss_fn()
                # Adaptive gamma
                gamma = adaptive_gamma.get_gamma(epoch, total_epochs)
                total_loss_val = task_loss + gamma * reg_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_val.backward()
            
            # Apply ARS damping
            damping = self.ars.step(task_loss.item(), self.training_config.get('learning_rate', 1e-5))
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad *= damping
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += task_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'damping': damping
            })
        
        avg_loss = total_loss / num_batches
        self.metrics['train_loss'].append(avg_loss)
        
        ars_stats = self.ars.get_statistics()
        if ars_stats:
            self.metrics['ars_damping'].append(ars_stats.get('ars_damping_avg', 1.0))
        
        return {
            'train_loss': avg_loss,
            'ars_stats': ars_stats
        }
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.metrics['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: str = './checkpoints'
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            checkpoint_dir: Directory for checkpoints
            
        Returns:
            Training results
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch, num_epochs)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )
            
            # Save checkpoint if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_file = checkpoint_path / f"model_best.pt"
                torch.save(self.model.state_dict(), checkpoint_file)
                logger.info(f"Saved best model to {checkpoint_file}")
        
        return {
            'best_val_loss': best_val_loss,
            'metrics': self.metrics
        }


def create_dummy_dataset(
    num_samples: int = 1000,
    seq_length: int = 512,
    vocab_size: int = 50257
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dummy dataset for testing.
    
    Args:
        num_samples: Number of samples
        seq_length: Sequence length
        vocab_size: Vocabulary size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dummy data
    inputs = torch.randint(0, vocab_size, (num_samples, seq_length))
    targets = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    # Split into train/val
    split = int(0.8 * num_samples)
    train_data = TensorDataset(inputs[:split], targets[:split])
    val_data = TensorDataset(inputs[split:], targets[split:])
    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)
    
    return train_loader, val_loader


def train_miniseries(
    start_model: int = 10,
    end_model: int = 12,
    config_path: Optional[str] = None
) -> None:
    """
    Train miniseries d10 → d11 → d12 → ...
    
    Args:
        start_model: Starting model (d10)
        end_model: Ending model (d20)
        config_path: Path to config file
    """
    logger.info(f"Starting miniseries training: d{start_model} → d{end_model}")
    
    # Model and training config
    model_config = {
        'vocab_size': 50257,
        'hidden_size': 768,
        'num_layers': 12
    }
    
    training_config = {
        'learning_rate': 1e-5,
        'weight_decay': 0.0,
        'regularization_gamma': 0.1,
        'ars_entropy_window': 10,
        'ars_surprise_window': 10,
        'batch_size': 8,
        'num_epochs': 3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create trainer
    trainer = MiniseriesTrainer(model_config, training_config)
    
    # Create dummy data
    train_loader, val_loader = create_dummy_dataset(num_samples=100)
    
    previous_weights = None
    
    # Train each model in series
    for model_idx in range(start_model, end_model + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training d{model_idx}")
        logger.info(f"{'='*60}")
        
        # Create model
        model = trainer.create_model(
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers']
        )
        
        # Initialize training
        trainer.initialize_training(model, previous_weights)
        
        # Train
        results = trainer.train(
            train_loader,
            val_loader,
            num_epochs=training_config['num_epochs'],
            checkpoint_dir=f'./checkpoints/d{model_idx}'
        )
        
        logger.info(f"d{model_idx} training complete - Best Val Loss: {results['best_val_loss']:.4f}")
        
        # Save weights for next model
        weights_path = f'./weights/d{model_idx}_weights.pt'
        Path('./weights').mkdir(exist_ok=True)
        save_model_weights(model, weights_path)
        
        # Load weights for next iteration
        previous_weights = load_anchor_weights(weights_path)
    
    logger.info("\nMiniseries training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train miniseries with progressive inheritance')
    parser.add_argument('--start', type=int, default=10, help='Starting model (d10)')
    parser.add_argument('--end', type=int, default=12, help='Ending model (d12)')
    parser.add_argument('--config', type=str, help='Config file path')
    
    args = parser.parse_args()
    
    train_miniseries(args.start, args.end, args.config)
