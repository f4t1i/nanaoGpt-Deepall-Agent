import json
import random
from typing import List, Dict, Any
from module_inventory import ModuleInventory
from reward_system import RewardSystem, ExecutionResult
from data_generator import TrainingDataGenerator

class ContinuousLearningTrainer:
    """Continuous Learning Trainer - learns from streaming data"""
    
    def __init__(self, inventory: ModuleInventory, learning_rate: float = 0.001):
        self.inventory = inventory
        self.learning_rate = learning_rate
        self.reward_system = RewardSystem()
        self.training_history = []
        self.model_state = {
            "total_samples_seen": 0,
            "total_reward": 0.0
        }
    
    def train_on_stream(self, num_batches: int = 10, batch_size: int = 32) -> Dict[str, Any]:
        """Train on streaming data with continuous updates"""
        generator = TrainingDataGenerator(self.inventory)
        all_modules = self.inventory.get_all_module_ids()
        
        results = {
            "total_batches": num_batches,
            "batch_size": batch_size,
            "batches": []
        }
        
        for batch_id in range(num_batches):
            batch_reward = 0.0
            batch_loss = 0.0
            batch_success = 0
            
            for sample_id in range(batch_size):
                # Generate streaming task
                expected = random.sample(all_modules, random.randint(1, 3))
                selected = random.sample(all_modules, random.randint(1, 3))
                
                execution_time = random.uniform(0.1, 2.0)
                success = len(set(selected) & set(expected)) > 0
                
                result = ExecutionResult(
                    task_id=f"batch{batch_id}_sample{sample_id}",
                    predicted_modules=selected,
                    actual_modules=expected,
                    execution_time=execution_time,
                    resource_usage=random.uniform(0.1, 1.0),
                    success=success
                )
                
                reward = self.reward_system.calculate_reward(result)
                batch_reward += reward
                batch_loss += abs(reward) * 0.01
                
                if success:
                    batch_success += 1
                
                # Update model state
                self.model_state["total_samples_seen"] += 1
                self.model_state["total_reward"] += reward
            
            avg_reward = batch_reward / batch_size
            avg_loss = batch_loss / batch_size
            success_rate = batch_success / batch_size
            
            batch_result = {
                "batch": batch_id + 1,
                "avg_reward": round(avg_reward, 4),
                "avg_loss": round(avg_loss, 4),
                "success_rate": round(success_rate, 4),
                "samples_processed": self.model_state["total_samples_seen"]
            }
            results["batches"].append(batch_result)
            self.training_history.append(batch_result)
            
            print(f"Batch {batch_id + 1}/{num_batches} - Reward: {avg_reward:.4f}, Loss: {avg_loss:.4f}, Success: {success_rate:.4f}")
        
        return results
    
    def adapt_to_new_task(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt model to a new task on-the-fly"""
        all_modules = self.inventory.get_all_module_ids()
        
        # Simulate task adaptation
        adaptation_steps = []
        
        for step in range(5):
            selected = random.sample(all_modules, random.randint(1, 3))
            expected = random.sample(all_modules, random.randint(1, 3))
            
            execution_time = random.uniform(0.1, 2.0)
            success = len(set(selected) & set(expected)) > 0
            
            result = ExecutionResult(
                task_id=f"adapt_step{step}",
                predicted_modules=selected,
                actual_modules=expected,
                execution_time=execution_time,
                resource_usage=random.uniform(0.1, 1.0),
                success=success
            )
            
            reward = self.reward_system.calculate_reward(result)
            
            adaptation_steps.append({
                "step": step + 1,
                "reward": round(reward, 4),
                "success": success
            })
        
        avg_adaptation_reward = sum(s['reward'] for s in adaptation_steps) / len(adaptation_steps)
        
        return {
            "task_context": task_context,
            "adaptation_steps": adaptation_steps,
            "avg_adaptation_reward": round(avg_adaptation_reward, 4)
        }
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get current model statistics"""
        total_samples = self.model_state["total_samples_seen"]
        total_reward = self.model_state["total_reward"]
        avg_reward = total_reward / total_samples if total_samples > 0 else 0.0
        
        return {
            "total_samples_seen": total_samples,
            "total_reward": round(total_reward, 4),
            "avg_reward_per_sample": round(avg_reward, 4)
        }
    
    def export_to_json(self, filename: str):
        """Export training history to JSON"""
        data = {
            "trainer": "ContinuousLearning",
            "learning_rate": self.learning_rate,
            "training_history": self.training_history,
            "model_state": self.model_state
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported Continuous Learning history to {filename}")

if __name__ == "__main__":
    inventory = ModuleInventory('deepall_modules.json')
    trainer = ContinuousLearningTrainer(inventory)
    
    # Train on streaming data
    results = trainer.train_on_stream(num_batches=5, batch_size=20)
    
    # Adapt to new task
    new_task = {"task_type": "classification", "complexity": "high"}
    adaptation_results = trainer.adapt_to_new_task(new_task)
    print(f"✓ Adaptation - Avg Reward: {adaptation_results['avg_adaptation_reward']:.4f}")
    
    # Get statistics
    stats = trainer.get_model_statistics()
    print(f"✓ Statistics - Samples: {stats['total_samples_seen']}, Avg Reward: {stats['avg_reward_per_sample']:.4f}")
    
    # Export
    trainer.export_to_json('continuous_learning_history.json')
