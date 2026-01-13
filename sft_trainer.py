import json
import random
from typing import List, Dict, Any
from module_inventory import ModuleInventory
from data_generator import TrainingDataGenerator

class SFTTrainer:
    """Supervised Fine-Tuning Trainer"""
    
    def __init__(self, inventory: ModuleInventory, learning_rate: float = 0.001):
        self.inventory = inventory
        self.learning_rate = learning_rate
        self.training_data = []
        self.training_history = []
    
    def generate_training_data(self, num_samples: int = 100):
        """Generate SFT training data"""
        generator = TrainingDataGenerator(self.inventory)
        self.training_data = generator.generate_dataset(num_samples=num_samples)
        print(f"✓ Generated {len(self.training_data)} SFT training examples")
    
    def train(self, num_epochs: int = 3, batch_size: int = 32) -> Dict[str, Any]:
        """Train the model using supervised fine-tuning"""
        if not self.training_data:
            raise ValueError("No training data. Call generate_training_data() first.")
        
        results = {
            "total_epochs": num_epochs,
            "total_samples": len(self.training_data),
            "batch_size": batch_size,
            "epochs": []
        }
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = (len(self.training_data) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(self.training_data))
                batch = self.training_data[start_idx:end_idx]
                
                # Simulate training step
                batch_loss = self._train_batch(batch)
                batch_accuracy = self._evaluate_batch(batch)
                
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
            
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            epoch_result = {
                "epoch": epoch + 1,
                "avg_loss": round(avg_loss, 4),
                "avg_accuracy": round(avg_accuracy, 4)
            }
            results["epochs"].append(epoch_result)
            self.training_history.append(epoch_result)
            
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        return results
    
    def _train_batch(self, batch: List) -> float:
        """Simulate training on a batch"""
        # Simulate loss calculation
        loss = random.uniform(0.1, 0.5)
        return loss
    
    def _evaluate_batch(self, batch: List) -> float:
        """Evaluate accuracy on a batch"""
        correct = 0
        for example in batch:
            # Check if selected modules match expected modules
            if set(example.selected_modules) == set(example.expected_modules):
                correct += 1
        
        return correct / len(batch) if batch else 0.0
    
    def evaluate(self, test_data: List = None) -> Dict[str, Any]:
        """Evaluate the model on test data"""
        if test_data is None:
            test_data = self.training_data
        
        correct = 0
        total = len(test_data)
        
        for example in test_data:
            if set(example.selected_modules) == set(example.expected_modules):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "total_samples": total,
            "correct": correct,
            "accuracy": round(accuracy, 4)
        }
    
    def export_to_json(self, filename: str):
        """Export training history to JSON"""
        data = {
            "trainer": "SFT",
            "learning_rate": self.learning_rate,
            "training_history": self.training_history,
            "num_training_samples": len(self.training_data)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported SFT training history to {filename}")

if __name__ == "__main__":
    inventory = ModuleInventory('deepall_modules.json')
    trainer = SFTTrainer(inventory)
    
    # Generate and train
    trainer.generate_training_data(num_samples=50)
    results = trainer.train(num_epochs=3, batch_size=10)
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"✓ Evaluation - Accuracy: {eval_results['accuracy']:.4f}")
    
    # Export
    trainer.export_to_json('sft_training_history.json')
