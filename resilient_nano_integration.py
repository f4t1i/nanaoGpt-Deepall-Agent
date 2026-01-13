#!/usr/bin/env python3
"""
Resilient-Nano-Trainer Integration with nanoGPT-DeepALL-Agent
Integrates ARS Optimizer with DeepALL Module Selection and Training
"""

import sys
import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime

# Import from Resilient-Nano-Trainer
sys.path.insert(0, '/home/ubuntu/Resilient-Nano-Trainer')
from ars_optimizer import ARSOptimizer

# Import from nanoGPT-DeepALL-Agent
from module_inventory import ModuleInventory
from enhanced_module_inventory import EnhancedModuleInventory
from deepall_integration_extended import DeepALLIntegrationExtended
from reward_system import RewardSystem, ExecutionResult


class ResilientNanoTrainingConfig:
    """Configuration for Resilient Nano Training"""
    
    def __init__(self):
        # ARS Optimizer parameters
        self.ars_alpha = 2.0
        self.ars_phi_min = 0.1
        self.ars_jitter_scale = 0.01
        self.ars_window_size = 50
        self.ars_rho_threshold = 0.7
        
        # Training parameters
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.batch_size = 32
        self.num_epochs = 10
        self.gradient_clip = 1.0
        
        # Module selection parameters
        self.num_modules = 5
        self.use_si_aware = True
        self.use_learning_aware = True
        self.use_performance_aware = True
        
        # Monitoring parameters
        self.log_interval = 10
        self.save_checkpoint_interval = 100
        self.enable_resilience_metrics = True


class ResilientNanoTrainer:
    """
    Integrates Resilient-Nano-Trainer with nanoGPT-DeepALL-Agent
    Provides ARS-stabilized training for DeepALL modules
    """
    
    def __init__(
        self,
        inventory: EnhancedModuleInventory,
        integration: DeepALLIntegrationExtended,
        config: Optional[ResilientNanoTrainingConfig] = None
    ):
        self.inventory = inventory
        self.integration = integration
        self.config = config or ResilientNanoTrainingConfig()
        self.reward_system = RewardSystem()
        
        # Training state
        self.training_history = []
        self.module_metrics = {}
        self.resilience_metrics = {
            'total_steps': 0,
            'recovery_events': 0,
            'divergence_prevented': 0,
            'avg_phi_t': 0.0,
            'avg_psi_t': 0.0,
            'avg_rho_1': 0.0
        }
        
        # ARS Optimizer (will be created per training session)
        self.optimizer = None
        self.ars_optimizer = None
    
    def select_optimal_modules(self, num_modules: int) -> List[str]:
        """
        Select optimal modules using DeepALL integration
        Considers SI, Learning, and Performance awareness
        """
        if self.config.use_si_aware:
            modules = self.integration.optimize_by_superintelligence(num_modules)
        elif self.config.use_learning_aware:
            modules = self.integration.optimize_for_learning(num_modules)
        elif self.config.use_performance_aware:
            modules = self.integration.optimize_for_performance(num_modules)
        else:
            modules = self.integration.optimize_module_selection(num_modules)
        
        return modules
    
    def create_ars_optimizer(self, model_params) -> Tuple[optim.Optimizer, ARSOptimizer]:
        """
        Create base optimizer and wrap with ARS
        """
        base_optimizer = optim.AdamW(
            model_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        ars_optimizer = ARSOptimizer(
            base_optimizer,
            alpha=self.config.ars_alpha,
            phi_min=self.config.ars_phi_min,
            jitter_scale=self.config.ars_jitter_scale,
            window_size=self.config.ars_window_size,
            rho_threshold=self.config.ars_rho_threshold
        )
        
        return base_optimizer, ars_optimizer
    
    def train_module(
        self,
        module_id: str,
        training_data: List[Dict],
        model: torch.nn.Module,
        loss_fn
    ) -> Dict[str, Any]:
        """
        Train a single module with ARS-stabilized optimization
        """
        module_info = self.inventory.get_module_enhanced(module_id)
        
        # Create ARS optimizer
        base_optimizer, ars_optimizer = self.create_ars_optimizer(model.parameters())
        self.optimizer = base_optimizer
        self.ars_optimizer = ars_optimizer
        
        # Training loop
        module_losses = []
        module_accuracies = []
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(training_data):
                # Forward pass
                logits = model(batch['input'])
                loss = loss_fn(logits, batch['target'])
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip
                    )
                
                # ARS-stabilized optimization step
                ars_optimizer.step(loss.item())
                
                # Track metrics
                epoch_loss += loss.item()
                num_batches += 1
                self.resilience_metrics['total_steps'] += 1
                
                # Update resilience metrics
                self._update_resilience_metrics(ars_optimizer)
                
                # Logging
                if batch_idx % self.config.log_interval == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"  Module {module_id} | Epoch {epoch+1}/{self.config.num_epochs} | "
                          f"Batch {batch_idx} | Loss: {avg_loss:.4f} | "
                          f"Φ_t: {ars_optimizer.phi_t:.3f} | Ψ_t: {ars_optimizer.psi_t:.3f}")
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            module_losses.append(avg_epoch_loss)
            
            print(f"  Module {module_id} | Epoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.4f}")
        
        # Calculate final metrics
        final_loss = module_losses[-1] if module_losses else 0.0
        avg_loss = sum(module_losses) / len(module_losses) if module_losses else 0.0
        
        # Create execution result for reward calculation
        execution_result = ExecutionResult(
            module_id=module_id,
            selected_modules=[module_id],
            executed_modules=[module_id],
            loss=final_loss,
            accuracy=1.0 - (final_loss / 10.0),  # Normalized accuracy
            success=final_loss < 5.0
        )
        
        # Calculate reward
        reward = self.reward_system.calculate_reward(execution_result)
        
        # Store training history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'module_id': module_id,
            'num_epochs': self.config.num_epochs,
            'final_loss': final_loss,
            'avg_loss': avg_loss,
            'reward': reward,
            'ars_metrics': {
                'final_phi_t': ars_optimizer.phi_t,
                'final_psi_t': ars_optimizer.psi_t,
                'final_rho_1': ars_optimizer.rho_1,
                'surprise_history_length': len(ars_optimizer.surprise_history)
            }
        }
        self.training_history.append(training_record)
        self.module_metrics[module_id] = training_record
        
        return {
            'module_id': module_id,
            'final_loss': final_loss,
            'avg_loss': avg_loss,
            'reward': reward,
            'num_epochs': self.config.num_epochs,
            'ars_metrics': training_record['ars_metrics'],
            'success': execution_result.success
        }
    
    def train_batch(
        self,
        module_ids: List[str],
        training_data: List[Dict],
        model: torch.nn.Module,
        loss_fn
    ) -> Dict[str, Any]:
        """
        Train multiple modules with ARS-stabilized optimization
        """
        batch_results = {}
        total_reward = 0.0
        successful_modules = 0
        
        print(f"\n{'='*80}")
        print(f"Training Batch: {len(module_ids)} modules")
        print(f"{'='*80}")
        
        for module_id in module_ids:
            print(f"\nTraining Module: {module_id}")
            try:
                result = self.train_module(module_id, training_data, model, loss_fn)
                batch_results[module_id] = result
                total_reward += result['reward']
                if result['success']:
                    successful_modules += 1
            except Exception as e:
                print(f"  ERROR: Failed to train module {module_id}: {str(e)}")
                batch_results[module_id] = {
                    'module_id': module_id,
                    'error': str(e),
                    'success': False
                }
        
        # Batch summary
        avg_reward = total_reward / len(module_ids) if module_ids else 0.0
        success_rate = successful_modules / len(module_ids) if module_ids else 0.0
        
        batch_summary = {
            'timestamp': datetime.now().isoformat(),
            'num_modules': len(module_ids),
            'successful_modules': successful_modules,
            'success_rate': success_rate,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'module_results': batch_results,
            'resilience_metrics': self.resilience_metrics.copy()
        }
        
        print(f"\n{'='*80}")
        print(f"Batch Summary:")
        print(f"  Modules Trained: {len(module_ids)}")
        print(f"  Successful: {successful_modules}/{len(module_ids)} ({success_rate*100:.1f}%)")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Total Steps: {self.resilience_metrics['total_steps']}")
        print(f"  Recovery Events: {self.resilience_metrics['recovery_events']}")
        print(f"{'='*80}\n")
        
        return batch_summary
    
    def _update_resilience_metrics(self, ars_optimizer: ARSOptimizer):
        """Update resilience metrics from ARS optimizer"""
        # Update running averages
        n = self.resilience_metrics['total_steps']
        
        self.resilience_metrics['avg_phi_t'] = (
            (self.resilience_metrics['avg_phi_t'] * (n - 1) + ars_optimizer.phi_t) / n
        )
        self.resilience_metrics['avg_psi_t'] = (
            (self.resilience_metrics['avg_psi_t'] * (n - 1) + ars_optimizer.psi_t) / n
        )
        self.resilience_metrics['avg_rho_1'] = (
            (self.resilience_metrics['avg_rho_1'] * (n - 1) + ars_optimizer.rho_1) / n
        )
        
        # Detect recovery events (when psi_t suddenly increases)
        if ars_optimizer.psi_t < 0.5 and len(ars_optimizer.loss_history) > 1:
            if ars_optimizer.loss_history[-1] < ars_optimizer.loss_history[-2]:
                self.resilience_metrics['recovery_events'] += 1
    
    def get_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_modules_trained': len(self.module_metrics),
            'total_training_steps': self.resilience_metrics['total_steps'],
            'recovery_events': self.resilience_metrics['recovery_events'],
            'avg_phi_t': self.resilience_metrics['avg_phi_t'],
            'avg_psi_t': self.resilience_metrics['avg_psi_t'],
            'avg_rho_1': self.resilience_metrics['avg_rho_1'],
            'module_metrics': self.module_metrics,
            'training_history': self.training_history,
            'config': {
                'ars_alpha': self.config.ars_alpha,
                'ars_phi_min': self.config.ars_phi_min,
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_epochs
            }
        }
    
    def save_report(self, filepath: str):
        """Save training report to JSON file"""
        report = self.get_training_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Training report saved: {filepath}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("Resilient-Nano-Trainer Integration with nanoGPT-DeepALL-Agent")
    print("="*80)
    
    # Initialize components
    print("\n[1/4] Initializing Module Inventory...")
    inventory = ModuleInventory('deepall_modules.json')
    print(f"  ✓ Loaded {len(inventory.modules)} modules")
    
    print("\n[2/4] Initializing Enhanced Inventory...")
    enhanced_inventory = EnhancedModuleInventory(inventory)
    print(f"  ✓ Enhanced inventory ready")
    
    print("\n[3/4] Initializing DeepALL Integration...")
    integration = DeepALLIntegrationExtended(enhanced_inventory)
    print(f"  ✓ DeepALL integration ready")
    
    print("\n[4/4] Initializing Resilient Nano Trainer...")
    config = ResilientNanoTrainingConfig()
    trainer = ResilientNanoTrainer(enhanced_inventory, integration, config)
    print(f"  ✓ Resilient Nano Trainer ready")
    
    # Select optimal modules
    print("\n[5/5] Selecting optimal modules...")
    optimal_modules = trainer.select_optimal_modules(5)
    print(f"  ✓ Selected modules: {optimal_modules}")
    
    print("\n" + "="*80)
    print("Integration successful! Ready for training.")
    print("="*80)
