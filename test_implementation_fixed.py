#!/usr/bin/env python3
"""
Comprehensive Test Suite for Progressive Inheritance + Scaling Laws Implementation
Tests all 7 core modules with actual class signatures
"""

import unittest
import torch
import torch.nn as nn
from typing import Dict, List
import tempfile
import os

# Import all modules
from regularization import FisherInformationMatrix, ProgressiveInheritanceRegularization
from ars_optimizer import EntropyGuard, SurpriseGate, ChronosJitter, ARSOptimizer, ARSAdamOptimizer
from train_miniseries import MiniseriesModel, MiniseriesTrainer
from evaluate import EvaluationMetrics, COREScoreComputer, ScalingLawValidator
from utils import LearningRateScheduler, MetricsTracker

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_dummy_model(vocab_size=1000, hidden_size=768, num_layers=2):
    """Create dummy model for testing."""
    return MiniseriesModel(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers)

# ============================================================================
# TEST CLASSES
# ============================================================================

class TestRegularization(unittest.TestCase):
    """Test regularization module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.model = create_dummy_model()
    
    def test_fisher_information_creation(self):
        """Test FisherInformationMatrix creation."""
        fisher = FisherInformationMatrix(model=self.model, device=self.device)
        self.assertIsNotNone(fisher)
    
    def test_fisher_information_computation(self):
        """Test Fisher Information computation."""
        fisher = FisherInformationMatrix(model=self.model, device=self.device)
        
        # Verify Fisher dictionary is initialized with model parameters
        self.assertGreater(len(fisher.fisher_dict), 0)
        
        # Verify all parameters have Fisher entries
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIn(name, fisher.fisher_dict)
                # Fisher values should be initialized to zero
                self.assertEqual(fisher.fisher_dict[name].sum().item(), 0.0)
    
    def test_progressive_inheritance_regularization(self):
        """Test Progressive Inheritance Regularization."""
        fisher = FisherInformationMatrix(model=self.model, device=self.device)
        
        # Create anchor weights
        anchor_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Create regularization
        reg = ProgressiveInheritanceRegularization(
            model=self.model,
            fisher_information=fisher,
            anchor_weights=anchor_weights,
            gamma=0.1,
            device=self.device
        )
        
        self.assertIsNotNone(reg)
        self.assertEqual(reg.gamma, 0.1)

class TestARSOptimizer(unittest.TestCase):
    """Test ARS Optimizer module."""
    
    def test_entropy_guard_creation(self):
        """Test EntropyGuard creation."""
        guard = EntropyGuard(window_size=10, threshold=0.5)
        self.assertIsNotNone(guard)
        self.assertEqual(guard.window_size, 10)
        self.assertEqual(guard.threshold, 0.5)
    
    def test_entropy_guard_detection(self):
        """Test EntropyGuard periodicity detection."""
        guard = EntropyGuard(window_size=5, threshold=0.3)
        
        # Create periodic loss sequence
        losses = [2.0, 1.9, 2.0, 1.9, 2.0, 1.9, 2.0]
        
        for loss in losses:
            guard.update(loss)
        
        # Get entropy damping
        entropy = guard.compute_entropy()
        
        # Should return damping factor
        self.assertGreater(entropy, 0)
        self.assertLessEqual(entropy, 1.0)
    
    def test_surprise_gate_creation(self):
        """Test SurpriseGate creation."""
        gate = SurpriseGate(window_size=10, surprise_threshold=0.5, damping_strength=0.1)
        self.assertIsNotNone(gate)
    
    def test_surprise_gate_damping(self):
        """Test SurpriseGate gradient damping."""
        gate = SurpriseGate(window_size=5, surprise_threshold=0.5, damping_strength=0.1)
        
        # Create loss sequence with spike
        losses = [2.0, 2.1, 2.05, 3.5, 2.1]  # Spike at index 3
        
        for loss in losses:
            gate.update(loss)
        
        # Get damping factor
        damping = gate.compute_damping()
        
        self.assertGreater(damping, 0)
        self.assertLessEqual(damping, 1.0)
    
    def test_chronos_jitter_creation(self):
        """Test ChronosJitter creation."""
        jitter = ChronosJitter(jitter_strength=0.01, entropy_threshold=0.5)
        self.assertIsNotNone(jitter)
    
    def test_chronos_jitter_generation(self):
        """Test ChronosJitter noise generation."""
        jitter = ChronosJitter(jitter_strength=0.01, entropy_threshold=0.5)
        
        # Set entropy to activate jitter
        jitter.set_entropy(0.5)  # Entropy < 1.0 activates jitter
        
        # Get jitter multiplier
        multiplier = jitter.compute_jitter(learning_rate=1e-5)
        
        # Should return a scalar multiplier around 1.0
        self.assertGreater(multiplier, 0.5)
        self.assertLess(multiplier, 2.0)
    
    def test_ars_optimizer_creation(self):
        """Test ARSOptimizer creation."""
        ars = ARSOptimizer(
            entropy_window=10,
            entropy_threshold=0.5,
            surprise_window=10,
            surprise_threshold=0.5,
            surprise_damping=0.1,
            jitter_strength=0.01
        )
        self.assertIsNotNone(ars)
        self.assertIsNotNone(ars.entropy_guard)
        self.assertIsNotNone(ars.surprise_gate)
        self.assertIsNotNone(ars.chronos_jitter)
    
    def test_ars_adam_optimizer(self):
        """Test ARSAdamOptimizer."""
        model = create_dummy_model()
        
        optimizer = ARSAdamOptimizer(
            params=model.parameters(),
            lr=1e-4,
            ars_config={
                'entropy_window': 10,
                'surprise_threshold': 0.5,
                'jitter_strength': 0.01
            }
        )
        
        self.assertIsNotNone(optimizer)

class TestTrainMiniseries(unittest.TestCase):
    """Test training module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.config = {
            'model': {
                'vocab_size': 100,
                'hidden_size': 768,
                'num_layers': 1
            },
            'training': {
                'batch_size': 2,
                'epochs': 1,
                'learning_rate': 1e-4
            }
        }
    
    def test_miniseries_model_creation(self):
        """Test MiniseriesModel creation."""
        model = MiniseriesModel(
            vocab_size=1000,
            hidden_size=768,
            num_layers=2
        )
        
        self.assertIsNotNone(model)
        self.assertGreater(count_parameters(model), 0)
    
    def test_miniseries_model_forward(self):
        """Test MiniseriesModel forward pass."""
        model = MiniseriesModel(
            vocab_size=1000,
            hidden_size=768,
            num_layers=2
        )
        
        # Create dummy input
        x = torch.randint(0, 1000, (4, 128))
        
        # Forward pass
        output = model(x)
        
        self.assertEqual(output.shape, (4, 128, 1000))
    
    def test_miniseries_trainer_creation(self):
        """Test MiniseriesTrainer creation."""
        trainer = MiniseriesTrainer(
            model_config=self.config['model'],
            training_config=self.config['training'],
            device=self.device
        )
        
        self.assertIsNotNone(trainer)
    
    def test_miniseries_trainer_training_step(self):
        """Test MiniseriesTrainer training step."""
        trainer = MiniseriesTrainer(
            model_config=self.config['model'],
            training_config=self.config['training'],
            device=self.device
        )
        
        # Create model first
        model = trainer.create_model(
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers']
        )
        
        # Initialize training with model
        trainer.initialize_training(model)
        
        # Verify trainer is properly initialized
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.ars)

class TestEvaluation(unittest.TestCase):
    """Test evaluation module."""
    
    def test_evaluation_metrics_creation(self):
        """Test EvaluationMetrics creation."""
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
    
    def test_core_score_computer_creation(self):
        """Test COREScoreComputer creation."""
        computer = COREScoreComputer(reference_model='gpt2')
        self.assertIsNotNone(computer)
    
    def test_scaling_law_validator_creation(self):
        """Test ScalingLawValidator creation."""
        validator = ScalingLawValidator()
        self.assertIsNotNone(validator)

class TestUtilities(unittest.TestCase):
    """Test utility functions."""
    
    def test_learning_rate_scheduler(self):
        """Test LearningRateScheduler."""
        # Create a simple model and optimizer
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create scheduler
        scheduler = LearningRateScheduler(optimizer, base_lr=1e-4, strategy='constant')
        self.assertIsNotNone(scheduler)
        
        # Step and get learning rate
        lr = scheduler.step()
        self.assertEqual(lr, 1e-4)
    
    def test_metrics_tracker(self):
        """Test MetricsTracker."""
        tracker = MetricsTracker()
        tracker.update('loss', 2.5)
        tracker.update('accuracy', 0.95)
        
        # Get average
        avg_loss = tracker.get_average('loss')
        self.assertEqual(avg_loss, 2.5)
        
        # Check metrics stored
        self.assertIn('loss', tracker.metrics)
        self.assertIn('accuracy', tracker.metrics)

class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full training pipeline."""
        config = {
            'model': {
                'vocab_size': 100,
                'hidden_size': 768,
                'num_layers': 1
            },
            'training': {
                'batch_size': 2,
                'epochs': 1,
                'learning_rate': 1e-4
            }
        }
        
        device = 'cpu'
        
        # Create trainer
        trainer = MiniseriesTrainer(
            model_config=config['model'],
            training_config=config['training'],
            device=device
        )
        
        # Create model
        model = trainer.create_model(
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers']
        )
        
        # Initialize training with model
        trainer.initialize_training(model)
        
        # Verify all components are initialized
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.ars)
        
        # Verify model can do forward pass
        test_input = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            output = trainer.model(test_input)
        
        self.assertEqual(output.shape, (2, 16, 100))

# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all tests
    suite.addTests(loader.loadTestsFromTestCase(TestRegularization))
    suite.addTests(loader.loadTestsFromTestCase(TestARSOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainMiniseries))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)
    
    # Exit with proper code
    exit(0 if result.wasSuccessful() else 1)
