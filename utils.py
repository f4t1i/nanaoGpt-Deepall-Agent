"""
Utility functions for miniseries training.
Includes configuration loading, data utilities, and helper functions.
"""

import torch
import torch.nn as nn
import yaml
import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved configuration to {output_path}")


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge override config into base config.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


# ============================================================================
# REPRODUCIBILITY UTILITIES
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed}")


# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def get_device(device_type: str = 'cuda', device_id: int = 0) -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device_type: "cuda" or "cpu"
        device_id: GPU device ID
        
    Returns:
        PyTorch device
    """
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def get_device_info() -> Dict[str, Any]:
    """
    Get device information.
    
    Returns:
        Dictionary with device info
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info['devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = 'INFO'
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Log file path
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def freeze_parameters(model: nn.Module) -> None:
    """
    Freeze all model parameters.
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_parameters(model: nn.Module) -> None:
    """
    Unfreeze all model parameters.
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: Dict,
    checkpoint_path: str
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        metrics: Training metrics
        checkpoint_path: Output path
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer,
    checkpoint_path: str,
    device: torch.device
) -> Tuple[int, Dict]:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        checkpoint_path: Checkpoint path
        device: Device to load to
        
    Returns:
        Tuple of (epoch, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    return epoch, metrics


# ============================================================================
# LEARNING RATE UTILITIES
# ============================================================================

class LearningRateScheduler:
    """Learning rate scheduler with various strategies."""
    
    def __init__(self, optimizer, base_lr: float, strategy: str = 'constant'):
        """
        Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            base_lr: Base learning rate
            strategy: Scheduling strategy
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.strategy = strategy
        self.step_count = 0
    
    def step(self) -> float:
        """
        Step scheduler and return current learning rate.
        
        Returns:
            Current learning rate
        """
        if self.strategy == 'constant':
            lr = self.base_lr
        elif self.strategy == 'linear_decay':
            lr = self.base_lr * (1 - self.step_count / 10000)
        elif self.strategy == 'exponential_decay':
            lr = self.base_lr * (0.95 ** (self.step_count / 100))
        elif self.strategy == 'cosine_annealing':
            lr = self.base_lr * (1 + np.cos(np.pi * self.step_count / 10000)) / 2
        else:
            lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.step_count += 1
        return lr


# ============================================================================
# METRICS UTILITIES
# ============================================================================

class MetricsTracker:
    """Track training metrics."""
    
    def __init__(self):
        """Initialize tracker."""
        self.metrics = {}
    
    def update(self, name: str, value: float) -> None:
        """
        Update metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_average(self, name: str, window: int = 100) -> float:
        """
        Get average metric over window.
        
        Args:
            name: Metric name
            window: Window size
            
        Returns:
            Average value
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        
        values = self.metrics[name][-window:]
        return np.mean(values)
    
    def save(self, output_path: str) -> None:
        """
        Save metrics to JSON.
        
        Args:
            output_path: Output file path
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {output_path}")


# ============================================================================
# DATA UTILITIES
# ============================================================================

def create_data_loaders(
    train_data,
    val_data,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple:
    """
    Create PyTorch data loaders.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Pin memory
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


# ============================================================================
# DIRECTORY UTILITIES
# ============================================================================

def create_directories(config: Dict) -> None:
    """
    Create necessary directories from config.
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config.get('checkpointing', {}).get('checkpoint_dir', './checkpoints'),
        config.get('checkpointing', {}).get('weights_dir', './weights'),
        config.get('logging', {}).get('tensorboard_dir', './tensorboard'),
        './logs',
        './reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created {len(directories)} directories")


if __name__ == '__main__':
    # Test utilities
    setup_logging()
    logger.info("Utils module loaded successfully")
    
    # Test device info
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Test config loading
    if Path('config.yaml').exists():
        config = load_config('config.yaml')
        logger.info(f"Loaded config with {len(config)} sections")
