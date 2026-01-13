import json
from typing import Dict, Any
from module_inventory import ModuleInventory
from sft_trainer import SFTTrainer
from rl_trainer import RLTrainer
from icl_trainer import ICLTrainer
from continuous_learning_trainer import ContinuousLearningTrainer
from deepall_integration import DeepALLIntegration

class TrainingOrchestrator:
    """Unified coordination of all 4 training methods"""
    
    def __init__(self, inventory: ModuleInventory):
        self.inventory = inventory
        self.sft_trainer = SFTTrainer(inventory)
        self.rl_trainer = RLTrainer(inventory)
        self.icl_trainer = ICLTrainer(inventory)
        self.cl_trainer = ContinuousLearningTrainer(inventory)
        self.deepall_integration = DeepALLIntegration(inventory)
        self.unified_results = {}
    
    def run_all_trainers(self, sft_samples: int = 50, rl_episodes: int = 5, 
                        icl_examples: int = 20, cl_batches: int = 5) -> Dict[str, Any]:
        """Run all 4 training methods sequentially"""
        
        print("\n" + "="*70)
        print("  Training Orchestrator: Running All 4 Methods")
        print("="*70 + "\n")
        
        # SFT Training
        print("[1/4] Running SFT (Supervised Fine-Tuning)...")
        self.sft_trainer.generate_training_data(num_samples=sft_samples)
        sft_results = self.sft_trainer.train(num_epochs=2, batch_size=10)
        sft_eval = self.sft_trainer.evaluate()
        print(f"✓ SFT Complete - Accuracy: {sft_eval['accuracy']:.4f}\n")
        
        # RL Training
        print("[2/4] Running RL (Reinforcement Learning)...")
        rl_results = self.rl_trainer.train(num_episodes=rl_episodes, steps_per_episode=10)
        rl_eval = self.rl_trainer.evaluate(num_test_episodes=2)
        print(f"✓ RL Complete - Success Rate: {rl_eval['success_rate']:.4f}\n")
        
        # ICL Training
        print("[3/4] Running ICL (In-Context Learning)...")
        self.icl_trainer.generate_training_data(num_examples=icl_examples, shots=3)
        icl_results = self.icl_trainer.train(num_iterations=2)
        icl_eval = self.icl_trainer.evaluate(num_test_examples=5)
        print(f"✓ ICL Complete - Accuracy: {icl_eval['avg_accuracy']:.4f}\n")
        
        # Continuous Learning Training
        print("[4/4] Running Continuous Learning...")
        cl_results = self.cl_trainer.train_on_stream(num_batches=cl_batches, batch_size=20)
        cl_stats = self.cl_trainer.get_model_statistics()
        print(f"✓ CL Complete - Avg Reward: {cl_stats['avg_reward_per_sample']:.4f}\n")
        
        # Compile unified results
        self.unified_results = {
            "sft": {
                "accuracy": sft_eval['accuracy'],
                "total_samples": sft_eval['total_samples'],
                "epochs": len(sft_results['epochs'])
            },
            "rl": {
                "success_rate": rl_eval['success_rate'],
                "avg_reward": rl_eval['avg_reward'],
                "episodes": rl_eval['test_episodes']
            },
            "icl": {
                "accuracy": icl_eval['avg_accuracy'],
                "test_examples": icl_eval['test_examples']
            },
            "cl": {
                "avg_reward": cl_stats['avg_reward_per_sample'],
                "total_samples": cl_stats['total_samples_seen']
            }
        }
        
        return self.unified_results
    
    def run_synergy_analysis(self) -> Dict[str, Any]:
        """Run synergy analysis on selected modules"""
        import random
        
        all_modules = self.inventory.get_all_module_ids()
        sample_modules = random.sample(all_modules, min(5, len(all_modules)))
        
        synergies = self.deepall_integration.detect_synergies(sample_modules)
        optimal = self.deepall_integration.optimize_module_selection(num_modules=5)
        
        return {
            "synergy_analysis": synergies,
            "optimization": optimal
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            "framework": "nanoGPT-DeepALL-Agent",
            "total_modules": len(self.inventory.modules),
            "training_results": self.unified_results,
            "synergy_analysis": self.run_synergy_analysis()
        }
        
        return report
    
    def export_results(self, filename: str = 'unified_training_results.json'):
        """Export unified results to JSON"""
        if not self.unified_results:
            print("Warning: No training results to export. Run run_all_trainers() first.")
            return
        
        data = {
            "framework": "nanoGPT-DeepALL-Agent",
            "total_modules": len(self.inventory.modules),
            "training_methods": 4,
            "results": self.unified_results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported unified results to {filename}")

if __name__ == "__main__":
    inventory = ModuleInventory('deepall_modules.json')
    orchestrator = TrainingOrchestrator(inventory)
    
    # Run all trainers
    results = orchestrator.run_all_trainers(
        sft_samples=30,
        rl_episodes=3,
        icl_examples=15,
        cl_batches=3
    )
    
    # Generate report
    report = orchestrator.generate_comprehensive_report()
    
    # Export
    orchestrator.export_results('unified_training_results.json')
    
    print("\n✓ Training Orchestrator Complete!")
    print(f"  SFT Accuracy: {results['sft']['accuracy']:.4f}")
    print(f"  RL Success Rate: {results['rl']['success_rate']:.4f}")
    print(f"  ICL Accuracy: {results['icl']['accuracy']:.4f}")
    print(f"  CL Avg Reward: {results['cl']['avg_reward']:.4f}")
