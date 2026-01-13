"""
Evaluation module for miniseries training.
Computes CORE scores, validates scaling laws, and generates reports.

CORE Score: Metric from DCLM paper for comparing models on same scale
Scaling Laws: Validates Chinchilla/nanoGPT scaling relationships
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    model_name: str
    num_parameters: int
    num_tokens: int
    perplexity: float
    core_score: float
    loss: float
    training_time_hours: float
    estimated_cost_usd: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class COREScoreComputer:
    """
    Compute CORE score from DCLM paper.
    
    CORE = Compute-Optimal Ranking Evaluation
    Allows comparing models trained with different compute budgets on same scale.
    """
    
    def __init__(self, reference_model: str = "gpt2"):
        """
        Initialize CORE score computer.
        
        Args:
            reference_model: Reference model for normalization ("gpt2", "gpt3", etc.)
        """
        self.reference_model = reference_model
        
        # Reference CORE scores from DCLM paper
        self.reference_scores = {
            "gpt2": {
                "perplexity": 29.41,
                "loss": 3.38,
                "num_parameters": 125000000,
                "num_tokens": 10000000000
            },
            "gpt3": {
                "perplexity": 20.5,
                "loss": 3.02,
                "num_parameters": 175000000,
                "num_tokens": 300000000000
            }
        }
    
    def compute_perplexity(self, loss: float) -> float:
        """
        Compute perplexity from loss.
        
        Perplexity = exp(loss)
        
        Args:
            loss: Cross-entropy loss
            
        Returns:
            Perplexity value
        """
        return np.exp(loss)
    
    def compute_core_score(
        self,
        loss: float,
        num_parameters: int,
        num_tokens: int
    ) -> float:
        """
        Compute CORE score.
        
        CORE normalizes loss by model size and training tokens to allow
        fair comparison across different compute budgets.
        
        Formula (simplified):
        CORE = loss × (num_parameters / ref_params)^α × (num_tokens / ref_tokens)^β
        
        Args:
            loss: Training loss
            num_parameters: Model size
            num_tokens: Training tokens
            
        Returns:
            CORE score
        """
        ref = self.reference_scores[self.reference_model]
        
        # Scaling exponents (from Chinchilla paper)
        alpha = 0.5  # Parameter scaling exponent
        beta = 0.5   # Token scaling exponent
        
        # Normalize by reference
        param_ratio = num_parameters / ref["num_parameters"]
        token_ratio = num_tokens / ref["num_tokens"]
        
        # Compute CORE score
        core_score = loss * (param_ratio ** alpha) * (token_ratio ** beta)
        
        return core_score
    
    def compute_chinchilla_ratio(
        self,
        num_parameters: int,
        num_tokens: int
    ) -> float:
        """
        Compute Chinchilla ratio (D/N).
        
        Optimal ratio is ~20 (Chinchilla paper) or ~8 (nanoGPT).
        
        Args:
            num_parameters: Model size
            num_tokens: Training tokens
            
        Returns:
            D/N ratio
        """
        return num_tokens / num_parameters


class ScalingLawValidator:
    """
    Validate scaling laws from Chinchilla and nanoGPT papers.
    """
    
    def __init__(self):
        """Initialize scaling law validator."""
        self.chinchilla_constant = 20.0  # Optimal D/N ratio
        self.nanogpt_constant = 8.0      # Empirical nanoGPT ratio
    
    def validate_chinchilla(
        self,
        losses: List[float],
        model_sizes: List[int],
        token_counts: List[int]
    ) -> Dict:
        """
        Validate Chinchilla scaling laws.
        
        Loss ∝ (C/N)^α × (C/D)^β
        where C = total compute, N = parameters, D = tokens
        
        Args:
            losses: List of training losses
            model_sizes: List of model sizes (parameters)
            token_counts: List of token counts
            
        Returns:
            Dictionary with validation results
        """
        if len(losses) < 2:
            return {"error": "Need at least 2 data points"}
        
        # Compute log ratios
        log_losses = np.log(losses)
        log_params = np.log(model_sizes)
        log_tokens = np.log(token_counts)
        
        # Fit power law: log(loss) = a + b*log(params) + c*log(tokens)
        X = np.column_stack([np.ones(len(losses)), log_params, log_tokens])
        coeffs = np.linalg.lstsq(X, log_losses, rcond=None)[0]
        
        # Extract exponents
        alpha = -coeffs[1]  # Parameter exponent (negative)
        beta = -coeffs[2]   # Token exponent (negative)
        
        # Compute R² for goodness of fit
        predictions = X @ coeffs
        ss_res = np.sum((log_losses - predictions) ** 2)
        ss_tot = np.sum((log_losses - np.mean(log_losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "r_squared": float(r_squared),
            "chinchilla_ratio": self.chinchilla_constant,
            "nanogpt_ratio": self.nanogpt_constant
        }
    
    def predict_loss(
        self,
        num_parameters: int,
        num_tokens: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        constant: float = 1.0
    ) -> float:
        """
        Predict loss using scaling law.
        
        Loss = constant × (num_params)^(-alpha) × (num_tokens)^(-beta)
        
        Args:
            num_parameters: Model size
            num_tokens: Training tokens
            alpha: Parameter exponent
            beta: Token exponent
            constant: Scaling constant
            
        Returns:
            Predicted loss
        """
        loss = constant * (num_parameters ** (-alpha)) * (num_tokens ** (-beta))
        return loss


class EvaluationReport:
    """
    Generate comprehensive evaluation reports.
    """
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_metrics_report(
        self,
        metrics_list: List[EvaluationMetrics],
        output_file: Optional[str] = None
    ) -> Dict:
        """
        Generate metrics report for miniseries.
        
        Args:
            metrics_list: List of evaluation metrics
            output_file: Output file path
            
        Returns:
            Report dictionary
        """
        report = {
            "timestamp": str(np.datetime64('now')),
            "num_models": len(metrics_list),
            "models": [m.to_dict() for m in metrics_list],
            "summary": self._compute_summary(metrics_list)
        }
        
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved metrics report to {output_path}")
        
        return report
    
    def _compute_summary(self, metrics_list: List[EvaluationMetrics]) -> Dict:
        """Compute summary statistics."""
        perplexities = [m.perplexity for m in metrics_list]
        losses = [m.loss for m in metrics_list]
        costs = [m.estimated_cost_usd for m in metrics_list]
        
        return {
            "avg_perplexity": float(np.mean(perplexities)),
            "min_perplexity": float(np.min(perplexities)),
            "max_perplexity": float(np.max(perplexities)),
            "avg_loss": float(np.mean(losses)),
            "min_loss": float(np.min(losses)),
            "max_loss": float(np.max(losses)),
            "total_cost_usd": float(np.sum(costs)),
            "perplexity_improvement": float((perplexities[0] - perplexities[-1]) / perplexities[0] * 100)
        }
    
    def plot_scaling_curves(
        self,
        metrics_list: List[EvaluationMetrics],
        output_file: str = "scaling_curves.png"
    ) -> None:
        """
        Plot scaling curves.
        
        Args:
            metrics_list: List of evaluation metrics
            output_file: Output file path
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        params = [m.num_parameters for m in metrics_list]
        tokens = [m.num_tokens for m in metrics_list]
        losses = [m.loss for m in metrics_list]
        perplexities = [m.perplexity for m in metrics_list]
        
        # Plot 1: Loss vs Parameters
        axes[0, 0].loglog(params, losses, 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Number of Parameters')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss vs Model Size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss vs Tokens
        axes[0, 1].loglog(tokens, losses, 's-', linewidth=2)
        axes[0, 1].set_xlabel('Number of Tokens')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss vs Training Tokens')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Perplexity vs Parameters
        axes[1, 0].loglog(params, perplexities, '^-', linewidth=2)
        axes[1, 0].set_xlabel('Number of Parameters')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].set_title('Perplexity vs Model Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: D/N Ratio
        ratios = [t / p for t, p in zip(tokens, params)]
        model_names = [f"d{10+i}" for i in range(len(metrics_list))]
        axes[1, 1].bar(model_names, ratios, color='steelblue')
        axes[1, 1].axhline(y=20, color='r', linestyle='--', label='Chinchilla (20)')
        axes[1, 1].axhline(y=8, color='g', linestyle='--', label='nanoGPT (8)')
        axes[1, 1].set_ylabel('D/N Ratio')
        axes[1, 1].set_title('Chinchilla Ratio (D/N)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scaling curves to {output_path}")
        plt.close()


def evaluate_model(
    model: nn.Module,
    val_loader,
    device: str = 'cuda'
) -> float:
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


if __name__ == '__main__':
    # Example usage
    logger.info("Evaluation module loaded successfully")
    
    # Create dummy metrics for testing
    metrics_list = [
        EvaluationMetrics(
            model_name=f"d{10+i}",
            num_parameters=7000000 * (2 ** i),
            num_tokens=56000000 * (2 ** i),
            perplexity=30.0 - i * 2,
            core_score=1.0 - i * 0.05,
            loss=3.4 - i * 0.1,
            training_time_hours=2.0 if i == 0 else 1.0,
            estimated_cost_usd=0.88 if i == 0 else 0.44
        )
        for i in range(5)
    ]
    
    # Generate report
    report_gen = EvaluationReport()
    report = report_gen.generate_metrics_report(metrics_list, "metrics_report.json")
    print(f"Generated report for {report['num_models']} models")
    
    # Validate scaling laws
    validator = ScalingLawValidator()
    params = [m.num_parameters for m in metrics_list]
    tokens = [m.num_tokens for m in metrics_list]
    losses = [m.loss for m in metrics_list]
    
    scaling_results = validator.validate_chinchilla(losses, params, tokens)
    print(f"Scaling law validation: {scaling_results}")
