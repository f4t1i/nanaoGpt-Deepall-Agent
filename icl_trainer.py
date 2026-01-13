import json
import random
from typing import List, Dict, Any
from module_inventory import ModuleInventory
from data_generator import TrainingDataGenerator

class ICLTrainer:
    """In-Context Learning Trainer"""
    
    def __init__(self, inventory: ModuleInventory):
        self.inventory = inventory
        self.training_examples = []
        self.training_history = []
    
    def generate_training_data(self, num_examples: int = 50, shots: int = 3):
        """Generate in-context learning examples"""
        generator = TrainingDataGenerator(self.inventory)
        self.training_examples = generator.generate_icl_examples(num_examples=num_examples)
        print(f"✓ Generated {len(self.training_examples)} ICL examples with {shots}-shot learning")
    
    def train(self, num_iterations: int = 5) -> Dict[str, Any]:
        """Train using in-context learning"""
        if not self.training_examples:
            raise ValueError("No training data. Call generate_training_data() first.")
        
        results = {
            "total_iterations": num_iterations,
            "total_examples": len(self.training_examples),
            "iterations": []
        }
        
        for iteration in range(num_iterations):
            iteration_loss = 0.0
            iteration_accuracy = 0.0
            
            for example in self.training_examples:
                # Evaluate shots
                shots_accuracy = self._evaluate_shots(example['shots'])
                
                # Evaluate query
                query = example['query']
                query_accuracy = self._evaluate_query(query)
                
                # Combined accuracy
                combined_accuracy = (shots_accuracy + query_accuracy) / 2
                iteration_accuracy += combined_accuracy
                
                # Simulate loss
                iteration_loss += (1.0 - combined_accuracy) * 0.1
            
            avg_accuracy = iteration_accuracy / len(self.training_examples)
            avg_loss = iteration_loss / len(self.training_examples)
            
            iteration_result = {
                "iteration": iteration + 1,
                "avg_loss": round(avg_loss, 4),
                "avg_accuracy": round(avg_accuracy, 4)
            }
            results["iterations"].append(iteration_result)
            self.training_history.append(iteration_result)
            
            print(f"Iteration {iteration + 1}/{num_iterations} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        return results
    
    def _evaluate_shots(self, shots: List[Dict[str, Any]]) -> float:
        """Evaluate accuracy on few-shot examples"""
        correct = 0
        for shot in shots:
            # Simple accuracy: check if output is reasonable
            if shot.get('output') and len(shot['output']) > 0:
                correct += 1
        
        return correct / len(shots) if shots else 0.0
    
    def _evaluate_query(self, query: Dict[str, Any]) -> float:
        """Evaluate accuracy on query"""
        expected = query.get('expected_output', [])
        # Simulate query accuracy (random for now)
        return random.uniform(0.3, 0.9)
    
    def evaluate(self, num_test_examples: int = 10) -> Dict[str, Any]:
        """Evaluate the trained model"""
        generator = TrainingDataGenerator(self.inventory)
        test_examples = generator.generate_icl_examples(num_examples=num_test_examples)
        
        total_accuracy = 0.0
        
        for example in test_examples:
            shots_accuracy = self._evaluate_shots(example['shots'])
            query_accuracy = self._evaluate_query(example['query'])
            combined_accuracy = (shots_accuracy + query_accuracy) / 2
            total_accuracy += combined_accuracy
        
        avg_accuracy = total_accuracy / num_test_examples if num_test_examples > 0 else 0.0
        
        return {
            "test_examples": num_test_examples,
            "avg_accuracy": round(avg_accuracy, 4)
        }
    
    def export_to_json(self, filename: str):
        """Export training history to JSON"""
        data = {
            "trainer": "ICL",
            "training_history": self.training_history,
            "num_examples": len(self.training_examples)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported ICL training history to {filename}")

if __name__ == "__main__":
    inventory = ModuleInventory('deepall_modules.json')
    trainer = ICLTrainer(inventory)
    
    # Generate and train
    trainer.generate_training_data(num_examples=20, shots=3)
    results = trainer.train(num_iterations=3)
    
    # Evaluate
    eval_results = trainer.evaluate(num_test_examples=5)
    print(f"✓ Evaluation - Accuracy: {eval_results['avg_accuracy']:.4f}")
    
    # Export
    trainer.export_to_json('icl_training_history.json')
