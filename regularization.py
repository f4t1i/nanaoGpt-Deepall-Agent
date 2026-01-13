"""
Regularization Module: Fisher Information Matrix for Progressive Model Inheritance
Based on Elastic Weight Consolidation (EWC) - Kirkpatrick et al. (2017)

This module implements Fisher Information-based regularization to prevent catastrophic
forgetting when training models sequentially (d10 → d11 → d12 → ... → d20).

Key Formula:
    L_total = L_new + (λ/2) Σ_i F_i (w_i - w*_i)²
    
where:
    L_new = loss on new task
    F_i = Fisher Information diagonal for parameter i
    w_i = current weights
    w*_i = previous task weights (anchor)
    λ = regularization strength (gamma)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class FisherInformationMatrix:
    """
    Computes and stores Fisher Information Matrix (diagonal approximation).
    
    The Fisher Information Matrix measures the importance of each parameter
    for the previous task. Parameters with high Fisher values are important
    and should not change much during new task training.
    
    Diagonal Approximation:
        F_i ≈ E[(∂L/∂w_i)²]  (squared gradients)
    
    This reduces computational complexity from O(n²) to O(n).
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize Fisher Information Matrix.
        
        Args:
            model: PyTorch model to compute Fisher for
            device: Device to compute on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.fisher_dict = {}
        self.param_names = []
        
        # Initialize Fisher dictionary with zero tensors
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.fisher_dict[name] = torch.zeros_like(param.data)
                self.param_names.append(name)
        
        logger.info(f"Initialized Fisher Information Matrix for {len(self.param_names)} parameters")
    
    def compute_fisher(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
        normalize: bool = True
    ) -> None:
        """
        Compute Fisher Information Matrix from data.
        
        Args:
            data_loader: DataLoader with previous task data
            num_batches: Number of batches to use (None = all)
            normalize: Whether to normalize by number of samples
        """
        self.model.eval()
        
        num_samples = 0
        batch_count = 0
        
        logger.info("Computing Fisher Information Matrix...")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc="Fisher Computation")):
                if num_batches and batch_idx >= num_batches:
                    break
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets, reduction='mean')
                
                # Compute gradients (without accumulating)
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                
                # Accumulate squared gradients
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.fisher_dict[name] += (param.grad ** 2).detach()
                
                num_samples += inputs.size(0)
                batch_count += 1
        
        # Normalize by number of samples
        if normalize and num_samples > 0:
            for name in self.fisher_dict:
                self.fisher_dict[name] /= num_samples
        
        logger.info(f"Fisher Information computed from {num_samples} samples ({batch_count} batches)")
    
    def get_fisher(self, name: str) -> torch.Tensor:
        """
        Get Fisher Information for a specific parameter.
        
        Args:
            name: Parameter name
            
        Returns:
            Fisher Information tensor
        """
        if name not in self.fisher_dict:
            raise ValueError(f"Parameter {name} not found in Fisher dictionary")
        return self.fisher_dict[name]
    
    def get_all_fisher(self) -> Dict[str, torch.Tensor]:
        """Get all Fisher Information matrices."""
        return self.fisher_dict.copy()
    
    def save(self, filepath: str) -> None:
        """Save Fisher Information to file."""
        torch.save(self.fisher_dict, filepath)
        logger.info(f"Fisher Information saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load Fisher Information from file."""
        self.fisher_dict = torch.load(filepath)
        logger.info(f"Fisher Information loaded from {filepath}")


class ProgressiveInheritanceRegularization:
    """
    Implements regularization loss for progressive model inheritance.
    
    When fine-tuning from a previous model, this regularization prevents
    large weight changes in important parameters (as measured by Fisher).
    
    Loss Term:
        L_reg = (λ/2) Σ_i F_i (w_i - w*_i)²
    
    where:
        F_i = Fisher Information for parameter i
        w_i = current weight
        w*_i = previous weight (anchor)
        λ = regularization strength
    """
    
    def __init__(
        self,
        model: nn.Module,
        fisher_information: FisherInformationMatrix,
        anchor_weights: Dict[str, torch.Tensor],
        gamma: float = 0.1,
        device: str = 'cuda'
    ):
        """
        Initialize Progressive Inheritance Regularization.
        
        Args:
            model: Current model being trained
            fisher_information: Fisher Information Matrix from previous task
            anchor_weights: Weights from previous task (w*_i)
            gamma: Regularization strength (λ)
            device: Device to compute on
        """
        self.model = model
        self.fisher = fisher_information
        self.anchor_weights = anchor_weights
        self.gamma = gamma
        self.device = device
        
        logger.info(f"Initialized Progressive Inheritance Regularization (γ={gamma})")
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss term.
        
        Returns:
            Scalar tensor containing regularization loss
        """
        reg_loss = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.anchor_weights:
                # Get Fisher Information for this parameter
                fisher = self.fisher.get_fisher(name)
                
                # Get anchor weight
                anchor = self.anchor_weights[name].to(self.device)
                
                # Compute weight change
                weight_change = param - anchor
                
                # Regularization: (λ/2) Σ_i F_i (w_i - w*_i)²
                reg_loss += (self.gamma / 2.0) * (fisher * (weight_change ** 2)).sum()
        
        return reg_loss
    
    def __call__(self) -> torch.Tensor:
        """Compute regularization loss (callable interface)."""
        return self.compute_regularization_loss()


class AdaptiveRegularizationStrength:
    """
    Adaptively adjusts regularization strength during training.
    
    Strategy:
        - Early epochs: Higher γ (preserve previous knowledge)
        - Later epochs: Lower γ (allow adaptation to new task)
    
    Formula:
        γ_t = γ_0 × (1 - t/T)^p
        
    where:
        γ_0 = initial strength
        t = current epoch
        T = total epochs
        p = decay power (default=1.0 for linear decay)
    """
    
    def __init__(self, gamma_0: float = 0.1, decay_power: float = 1.0):
        """
        Initialize Adaptive Regularization Strength.
        
        Args:
            gamma_0: Initial regularization strength
            decay_power: Decay power (1.0=linear, 2.0=quadratic)
        """
        self.gamma_0 = gamma_0
        self.decay_power = decay_power
    
    def get_gamma(self, epoch: int, total_epochs: int) -> float:
        """
        Get regularization strength for current epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            Regularization strength γ_t
        """
        if total_epochs <= 1:
            return self.gamma_0
        
        progress = epoch / total_epochs
        decay_factor = (1.0 - progress) ** self.decay_power
        gamma_t = self.gamma_0 * decay_factor
        
        return gamma_t


class CombinedRegularizationLoss:
    """
    Combines multiple regularization terms for training.
    
    Total Loss:
        L_total = L_task + w_reg × L_reg + w_l2 × L_l2
        
    where:
        L_task = task loss (cross-entropy, etc.)
        L_reg = progressive inheritance regularization
        L_l2 = standard L2 regularization
        w_reg, w_l2 = weights for each term
    """
    
    def __init__(
        self,
        task_loss_fn: callable,
        reg_loss_fn: Optional[callable] = None,
        weight_reg: float = 1.0,
        weight_l2: float = 0.0,
        l2_strength: float = 1e-5
    ):
        """
        Initialize Combined Regularization Loss.
        
        Args:
            task_loss_fn: Task loss function (e.g., cross_entropy)
            reg_loss_fn: Regularization loss function (progressive inheritance)
            weight_reg: Weight for regularization term
            weight_l2: Weight for L2 regularization
            l2_strength: L2 regularization strength
        """
        self.task_loss_fn = task_loss_fn
        self.reg_loss_fn = reg_loss_fn
        self.weight_reg = weight_reg
        self.weight_l2 = weight_l2
        self.l2_strength = l2_strength
    
    def __call__(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs
            targets: Target labels
            model: Model (for L2 regularization)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Task loss
        task_loss = self.task_loss_fn(outputs, targets)
        
        # Regularization loss
        reg_loss = torch.tensor(0.0, device=outputs.device)
        if self.reg_loss_fn is not None:
            reg_loss = self.reg_loss_fn()
        
        # L2 regularization
        l2_loss = torch.tensor(0.0, device=outputs.device)
        if self.weight_l2 > 0:
            for param in model.parameters():
                if param.requires_grad:
                    l2_loss += (param ** 2).sum()
            l2_loss = self.l2_strength * l2_loss
        
        # Combined loss
        total_loss = (
            task_loss +
            self.weight_reg * reg_loss +
            self.weight_l2 * l2_loss
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'task_loss': task_loss.item(),
            'reg_loss': reg_loss.item() if reg_loss.item() > 0 else 0.0,
            'l2_loss': l2_loss.item() if l2_loss.item() > 0 else 0.0,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


# Utility Functions

def save_model_weights(model: nn.Module, filepath: str) -> None:
    """
    Save model weights as anchor for next training phase.
    
    Args:
        model: Model to save
        filepath: Path to save weights
    """
    weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights[name] = param.data.clone().detach()
    torch.save(weights, filepath)
    logger.info(f"Model weights saved to {filepath}")


def load_anchor_weights(filepath: str) -> Dict[str, torch.Tensor]:
    """
    Load anchor weights from previous training phase.
    
    Args:
        filepath: Path to anchor weights
        
    Returns:
        Dictionary of anchor weights
    """
    weights = torch.load(filepath)
    logger.info(f"Anchor weights loaded from {filepath}")
    return weights


def compute_weight_change_statistics(
    model: nn.Module,
    anchor_weights: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute statistics on weight changes from anchor.
    
    Args:
        model: Current model
        anchor_weights: Anchor weights from previous phase
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'max_change': 0.0,
        'mean_change': 0.0,
        'num_params': 0,
        'num_changed': 0
    }
    
    total_change = 0.0
    
    for name, param in model.named_parameters():
        if param.requires_grad and name in anchor_weights:
            anchor = anchor_weights[name]
            change = (param.data - anchor).abs()
            
            stats['max_change'] = max(stats['max_change'], change.max().item())
            total_change += change.sum().item()
            stats['num_params'] += param.numel()
            
            if change.max().item() > 1e-6:
                stats['num_changed'] += 1
    
    if stats['num_params'] > 0:
        stats['mean_change'] = total_change / stats['num_params']
    
    return stats
