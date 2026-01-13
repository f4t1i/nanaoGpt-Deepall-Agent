import json
import random
from typing import List, Dict, Any
from module_inventory import ModuleInventory

class TrainingExample:
    def __init__(self, task_id: str, context: str, selected_modules: List[str], expected_modules: List[str]):
        self.task_id = task_id
        self.context = context
        self.selected_modules = selected_modules
        self.expected_modules = expected_modules
    
    def to_dict(self):
        return {
            "task_id": self.task_id,
            "context": self.context,
            "selected_modules": self.selected_modules,
            "expected_modules": self.expected_modules
        }

class TrainingDataGenerator:
    def __init__(self, inventory: ModuleInventory):
        self.inventory = inventory
        self.all_modules = self.inventory.get_all_module_ids()
    
    def generate_dataset(self, num_samples: int = 100) -> List[TrainingExample]:
        """Generate synthetic training data for SFT"""
        dataset = []
        
        for i in range(num_samples):
            task_id = f"task_{i:04d}"
            
            # Randomly select 1-3 expected modules
            num_expected = random.randint(1, 3)
            expected_modules = random.sample(self.all_modules, min(num_expected, len(self.all_modules)))
            
            # Create context from selected modules
            context = self._create_context_from_modules(expected_modules)
            
            # Selected modules (with some noise for realism)
            if random.random() < 0.7:  # 70% correct
                selected_modules = expected_modules
            else:  # 30% with errors
                selected_modules = expected_modules.copy()
                if random.random() < 0.5 and len(self.all_modules) > len(expected_modules):
                    # Add wrong module
                    wrong = random.choice([m for m in self.all_modules if m not in selected_modules])
                    selected_modules.append(wrong)
                elif selected_modules:
                    # Remove correct module
                    selected_modules.pop()
            
            example = TrainingExample(task_id, context, selected_modules, expected_modules)
            dataset.append(example)
        
        return dataset
    
    def _create_context_from_modules(self, module_ids: List[str]) -> str:
        """Create a descriptive context from module information"""
        modules = [self.inventory.get_module(mid) for mid in module_ids if self.inventory.get_module(mid)]
        
        if not modules:
            return "Generic task requiring module selection"
        
        categories = set(m.category for m in modules)
        methods = set(m.ai_training_method for m in modules)
        
        context = f"Task requiring modules from {', '.join(categories)}. "
        context += f"AI methods: {', '.join(methods)}. "
        context += f"Modules: {', '.join(m.name for m in modules[:2])}"
        
        return context
    
    def generate_rl_episodes(self, num_episodes: int = 10) -> List[Dict[str, Any]]:
        """Generate episodes for RL training"""
        episodes = []
        
        for ep_id in range(num_episodes):
            num_steps = random.randint(5, 15)
            steps = []
            
            for step_id in range(num_steps):
                # Random module selection
                selected = random.sample(self.all_modules, random.randint(1, 3))
                expected = random.sample(self.all_modules, random.randint(1, 3))
                
                steps.append({
                    "step": step_id,
                    "selected": selected,
                    "expected": expected,
                    "reward": 0.0  # Will be calculated by reward system
                })
            
            episodes.append({
                "episode_id": f"ep_{ep_id:04d}",
                "steps": steps
            })
        
        return episodes
    
    def generate_icl_examples(self, num_examples: int = 50) -> List[Dict[str, Any]]:
        """Generate in-context learning examples"""
        examples = []
        
        for i in range(num_examples):
            # Create few-shot examples
            num_shots = random.randint(2, 5)
            shots = []
            
            for shot_id in range(num_shots):
                selected = random.sample(self.all_modules, random.randint(1, 2))
                expected = random.sample(self.all_modules, random.randint(1, 2))
                
                shots.append({
                    "input": self._create_context_from_modules(selected),
                    "output": expected
                })
            
            # Query example
            query_selected = random.sample(self.all_modules, random.randint(1, 2))
            query_expected = random.sample(self.all_modules, random.randint(1, 2))
            
            examples.append({
                "example_id": f"icl_{i:04d}",
                "shots": shots,
                "query": {
                    "input": self._create_context_from_modules(query_selected),
                    "expected_output": query_expected
                }
            })
        
        return examples

if __name__ == "__main__":
    inventory = ModuleInventory('deepall_modules.json')
    generator = TrainingDataGenerator(inventory)
    
    # Generate SFT dataset
    sft_data = generator.generate_dataset(num_samples=10)
    print(f"✓ Generated {len(sft_data)} SFT examples")
    
    # Generate RL episodes
    rl_episodes = generator.generate_rl_episodes(num_episodes=5)
    print(f"✓ Generated {len(rl_episodes)} RL episodes")
    
    # Generate ICL examples
    icl_examples = generator.generate_icl_examples(num_examples=10)
    print(f"✓ Generated {len(icl_examples)} ICL examples")
