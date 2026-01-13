import json
import random
from typing import List, Dict, Any
from module_inventory import ModuleInventory
from reward_system import RewardSystem, ExecutionResult
from data_generator import TrainingDataGenerator

class RLTrainer:
    """Reinforcement Learning Trainer"""
    
    def __init__(self, inventory: ModuleInventory, learning_rate: float = 0.001):
        self.inventory = inventory
        self.learning_rate = learning_rate
        self.reward_system = RewardSystem()
        self.training_history = []
    
    def train(self, num_episodes: int = 10, steps_per_episode: int = 20) -> Dict[str, Any]:
        """Train using reinforcement learning"""
        generator = TrainingDataGenerator(self.inventory)
        all_modules = self.inventory.get_all_module_ids()
        
        results = {
            "total_episodes": num_episodes,
            "steps_per_episode": steps_per_episode,
            "episodes": []
        }
        
        for episode_id in range(num_episodes):
            episode_rewards = []
            episode_loss = 0.0
            
            for step_id in range(steps_per_episode):
                # Generate random task
                expected = random.sample(all_modules, random.randint(1, 3))
                
                # Agent selects modules (random for now)
                selected = random.sample(all_modules, random.randint(1, 3))
                
                # Calculate reward
                execution_time = random.uniform(0.1, 2.0)
                success = len(set(selected) & set(expected)) > 0
                
                result = ExecutionResult(
                    task_id=f"ep{episode_id}_step{step_id}",
                    selected_modules=selected,
                    expected_modules=expected,
                    execution_time=execution_time,
                    efficiency_score=random.uniform(0.1, 1.0),
                    success=success
                )
                
                reward = self.reward_system.calculate_reward(result)
                episode_rewards.append(reward)
                
                # Simulate loss update
                episode_loss += abs(reward) * 0.01
            
            avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
            
            episode_result = {
                "episode": episode_id + 1,
                "avg_reward": round(avg_reward, 4),
                "total_loss": round(episode_loss, 4),
                "num_steps": steps_per_episode
            }
            results["episodes"].append(episode_result)
            self.training_history.append(episode_result)
            
            print(f"Episode {episode_id + 1}/{num_episodes} - Avg Reward: {avg_reward:.4f}, Loss: {episode_loss:.4f}")
        
        return results
    
    def evaluate(self, num_test_episodes: int = 5) -> Dict[str, Any]:
        """Evaluate the trained policy"""
        generator = TrainingDataGenerator(self.inventory)
        all_modules = self.inventory.get_all_module_ids()
        
        total_reward = 0.0
        total_success = 0
        
        for ep_id in range(num_test_episodes):
            for step_id in range(10):
                expected = random.sample(all_modules, random.randint(1, 3))
                selected = random.sample(all_modules, random.randint(1, 3))
                
                execution_time = random.uniform(0.1, 2.0)
                success = len(set(selected) & set(expected)) > 0
                
                result = ExecutionResult(
                    task_id=f"test_ep{ep_id}_step{step_id}",
                    selected_modules=selected,
                    expected_modules=expected,
                    execution_time=execution_time,
                    efficiency_score=random.uniform(0.1, 1.0),
                    success=success
                )
                
                reward = self.reward_system.calculate_reward(result)
                total_reward += reward
                if success:
                    total_success += 1
        
        total_steps = num_test_episodes * 10
        avg_reward = total_reward / total_steps if total_steps > 0 else 0.0
        success_rate = total_success / total_steps if total_steps > 0 else 0.0
        
        return {
            "test_episodes": num_test_episodes,
            "total_steps": total_steps,
            "avg_reward": round(avg_reward, 4),
            "success_rate": round(success_rate, 4)
        }
    
    def export_training_history(self, filename: str):
        """Export training history to JSON"""
        data = {
            "trainer": "RL",
            "learning_rate": self.learning_rate,
            "training_history": self.training_history,
            "num_episodes": len(self.training_history)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported RL training history to {filename}")

if __name__ == "__main__":
    inventory = ModuleInventory('deepall_modules.json')
    trainer = RLTrainer(inventory)
    
    # Train
    results = trainer.train(num_episodes=5, steps_per_episode=10)
    
    # Evaluate
    eval_results = trainer.evaluate(num_test_episodes=3)
    print(f"✓ Evaluation - Success Rate: {eval_results['success_rate']:.4f}")
    
    # Export
    trainer.export_training_history('rl_training_history.json')
