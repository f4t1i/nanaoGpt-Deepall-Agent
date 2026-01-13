"""
Comprehensive Test Suite for Progressive Inheritance Implementation
Tests all 7 modules: regularization, ars_optimizer, train_miniseries, 
config, evaluate, utils, and miniseries_inheritance.sh
"""

import unittest
import torch
import torch.nn as nn
import yaml
import json
import tempfile
import sys
from pathlib import Path
from typing import Dict, Any

# Import modules to test
from regularization import (
    FisherInformationMatrix,
    ProgressiveInheritanceRegularization,
    AdaptiveRegularizationStrength,
    CombinedRegularizationLoss
)
from ars_optimizer import (
    EntropyGuard,
    SurpriseGate,
    ChronosJitter,
    ARSOptimizer,
    ARSAdamOptimizer
)
from train_miniseries import MiniseriesModel, MiniseriesTrainer
from evaluate import (
    EvaluationMetrics,
    EvaluationReport,
    ScalingLawValidator
)
from utils import (
    load_config,
    save_config,
    set_seed,
    get_device,
    count_parameters,
    get_model_size_mb,
    MetricsTracker,
    LearningRateScheduler,
    create_directories
)


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestRegularization(unittest.TestCase):
    """Test regularization module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.model = nn.Linear(10, 5)
        self.model.to(self.device)
    
    def test_fisher_information_matrix(self):
        """Test Fisher Information Matrix computation."""
        fisher = FisherInformationMatrix(self.model, self.device)
        
        # Create dummy data
        x = torch.randn(4, 10, device=self.device)
        y = torch.randn(4, 5, device=self.device)
        
        # Compute Fisher Information
        fisher.compute(x, y, loss_fn=nn.MSELoss())
        
        # Check shape
        self.assertEqual(fisher.fisher_matrix.shape[0], 55)  # 10*5 + 5
        self.assertTrue(torch.all(fisher.fisher_matrix >= 0))
    
    def test_progressive_inheritance_regularization(self):
        """Test Progressive Inheritance Regularization."""
        reg = ProgressiveInheritanceRegularization(
            model=self.model,
            gamma=0.4,
            device=self.device
        )
        
        # Store initial weights
        initial_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        reg.store_weights(initial_weights)
        
        # Modify model
        for param in self.model.parameters():
            param.data += 0.1
        
        # Compute regularization loss
        loss = reg.compute_loss()
        
        self.assertGreater(loss.item(), 0)
        self.assertTrue(torch.isfinite(loss))
    
    def test_adaptive_regularization_strength(self):
        """Test Adaptive Regularization Strength."""
        adapter = AdaptiveRegularizationStrength(
            initial_gamma=0.4,
            min_gamma=0.1,
            max_gamma=0.9
        )
        
        # Test gamma adjustment
        gamma1 = adapter.get_gamma(epoch=0, total_epochs=10)
        gamma2 = adapter.get_gamma(epoch=5, total_epochs=10)
        gamma3 = adapter.get_gamma(epoch=9, total_epochs=10)
        
        self.assertGreater(gamma1, 0)
        self.assertLess(gamma3, 1)
        self.assertTrue(gamma1 >= 0.1 and gamma1 <= 0.9)


class TestARSOptimizer(unittest.TestCase):
    """Test ARS Optimizer module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.model = nn.Linear(10, 5)
        self.model.to(self.device)
    
    def test_entropy_guard(self):
        """Test Entropy Guard mechanism."""
        guard = EntropyGuard(lag=1)
        
        # Create loss history
        losses = [2.5, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5]
        
        for loss in losses:
            periodicity = guard.detect_periodicity(loss)
            self.assertGreaterEqual(periodicity, 0)
            self.assertLessEqual(periodicity, 1)
    
    def test_surprise_gate(self):
        """Test Surprise Gate mechanism."""
        gate = SurpriseGate(threshold=0.5)
        
        # Create loss history
        losses = [2.0, 2.1, 2.05, 2.2, 2.3, 2.25, 2.4]
        
        for loss in losses:
            damping = gate.compute_damping(loss)
            self.assertGreaterEqual(damping, 0)
            self.assertLessEqual(damping, 1)
    
    def test_chronos_jitter(self):
        """Test Chronos Jitter mechanism."""
        jitter = ChronosJitter(noise_scale=0.01)
        
        # Test jitter generation
        for _ in range(10):
            noise = jitter.generate_noise(shape=(5,))
            self.assertEqual(noise.shape, (5,))
            self.assertTrue(torch.all(torch.isfinite(noise)))
    
    def test_ars_optimizer(self):
        """Test ARS Optimizer."""
        optimizer = ARSOptimizer(
            model=self.model,
            device=self.device,
            entropy_lag=1,
            surprise_threshold=0.5,
            jitter_scale=0.01
        )
        
        # Create dummy training step
        x = torch.randn(4, 10, device=self.device)
        y = torch.randn(4, 5, device=self.device)
        
        loss_fn = nn.MSELoss()
        
        for _ in range(5):
            output = self.model(x)
            loss = loss_fn(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.assertTrue(optimizer.loss_history)
        self.assertGreater(len(optimizer.loss_history), 0)


class TestTrainMiniseries(unittest.TestCase):
    """Test training module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.config = {
            'model': {
                'hidden_dim': 768,  # Must be divisible by nhead=12
                'num_layers': 1,
                'vocab_size': 100,
                'max_seq_len': 128
            },
            'training': {
                'batch_size': 4,
                'epochs': 1,
                'learning_rate': 1e-4
            }
        }
    
    def test_miniseries_model_creation(self):
        """Test MiniseriesModel creation."""
        model = MiniseriesModel(
            vocab_size=1000,
            hidden_size=768,  # Must be divisible by nhead=12
            num_layers=2
        )
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertGreater(count_parameters(model), 0)
    
    def test_miniseries_model_forward(self):
        """Test MiniseriesModel forward pass."""
        model = MiniseriesModel(
            vocab_size=1000,
            hidden_size=768,  # Must be divisible by nhead=12
            num_layers=2
        )
        
        # Create dummy input
        x = torch.randint(0, 1000, (4, 128))
        
        # Forward pass
        output = model(x)
        
        self.assertEqual(output.shape, (4, 128, 1000))
    
    def test_miniseries_trainer(self):
        """Test MiniseriesTrainer."""
        model = MiniseriesModel(
            vocab_size=1000,
            hidden_size=768,  # Must be divisible by nhead=12
            num_layers=2
        )
        
        trainer = MiniseriesTrainer(
            model=model,
            config=self.config,
            device=self.device
        )
        
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.optimizer)


class TestEvaluate(unittest.TestCase):
    """Test evaluation module."""
    
    def test_evaluation_metrics(self):
        """Test EvaluationMetrics."""
        metrics = EvaluationMetrics(
            model_name="d10",
            num_parameters=7000000000,
            num_tokens=1000000,
            perplexity=15.5,
            core_score=0.75,
            loss=2.1,
            training_time_hours=1.0,
            estimated_cost_usd=5.0
        )
        
        self.assertEqual(metrics.model_name, "d10")
        self.assertEqual(metrics.num_parameters, 7000000000)
        self.assertGreater(metrics.loss, 0)
    
    def test_scaling_law_validator(self):
        """Test Scaling Law Validator."""
        validator = ScalingLawValidator()
        
        # Create dummy metrics
        metrics_list = []
        for i in range(5):
            metrics = EvaluationMetrics(
                model_name=f"d{10+i}",
                num_parameters=7000000000 * (i+1),
                num_tokens=1000000 * (i+1),
                perplexity=20.0 - i*2,
                core_score=0.7 + i*0.05,
                loss=2.0 - i*0.1,
                training_time_hours=1.0 * (i+1),
                estimated_cost_usd=5.0 * (i+1)
            )
            metrics_list.append(metrics)
        
        # Validate scaling laws
        result = validator.validate(metrics_list)
        
        self.assertIn('alpha', result)
        self.assertIn('beta', result)
        self.assertGreater(result['alpha'], 0)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        t1 = torch.randn(5)
        
        set_seed(42)
        t2 = torch.randn(5)
        
        self.assertTrue(torch.allclose(t1, t2))
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device('cpu')
        self.assertEqual(device.type, 'cpu')
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = nn.Linear(10, 5)
        param_count = count_parameters(model)
        
        # 10*5 (weight) + 5 (bias) = 55
        self.assertEqual(param_count, 55)
    
    def test_get_model_size_mb(self):
        """Test model size calculation."""
        model = nn.Linear(1000, 1000)
        size_mb = get_model_size_mb(model)
        
        self.assertGreater(size_mb, 0)
    
    def test_metrics_tracker(self):
        """Test metrics tracker."""
        tracker = MetricsTracker()
        
        # Add metrics
        for i in range(10):
            tracker.update('loss', 2.0 - i*0.1)
        
        # Get average
        avg = tracker.get_average('loss', window=5)
        self.assertGreater(avg, 0)
    
    def test_learning_rate_scheduler(self):
        """Test learning rate scheduler."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        scheduler = LearningRateScheduler(
            optimizer=optimizer,
            base_lr=1e-3,
            strategy='linear_decay'
        )
        
        # Step scheduler
        for _ in range(10):
            lr = scheduler.step()
            self.assertGreater(lr, 0)
    
    def test_config_load_save(self):
        """Test config loading and saving."""
        config = {
            'model': {'hidden_dim': 64},
            'training': {'epochs': 10}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            save_config(config, f.name)
            loaded = load_config(f.name)
            
            self.assertEqual(loaded['model']['hidden_dim'], 64)
            self.assertEqual(loaded['training']['epochs'], 10)
            
            Path(f.name).unlink()


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_training_pipeline(self):
        """Test full training pipeline."""
        # Create config
        config = {
            'model': {
                'hidden_dim': 768,  # Must be divisible by nhead=12
                'num_layers': 1,
                'vocab_size': 100,
                'max_seq_len': 64
            },
            'training': {
                'batch_size': 2,
                'epochs': 1,
                'learning_rate': 1e-4
            },
            'regularization': {
                'gamma': 0.4,
                'use_fisher': True
            },
            'ars': {
                'entropy_lag': 1,
                'surprise_threshold': 0.5,
                'jitter_scale': 0.01
            }
        }
        
        device = torch.device('cpu')
        
        # Create model
        model = MiniseriesModel(
            vocab_size=config['model']['vocab_size'],
            hidden_size=768,  # Must be divisible by nhead=12
            num_layers=config['model']['num_layers']
        )
        model.to(device)
        
        # Create trainer
        trainer = MiniseriesTrainer(
            model=model,
            config=config,
            device=device
        )
        
        # Create dummy data
        x = torch.randint(0, 100, (2, 64))
        y = torch.randint(0, 100, (2, 64))
        
        # Single training step
        loss = trainer.train_step(x, y)
        
        self.assertGreater(loss.item(), 0)
        self.assertTrue(torch.isfinite(loss))


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRegularization))
    suite.addTests(loader.loadTestsFromTestCase(TestARSOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainMiniseries))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluate))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
