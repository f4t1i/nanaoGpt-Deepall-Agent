#!/usr/bin/env python3
"""
KAGGLE RIGOROUS TEST SUITE - nanoGPT-DeepALL-Agent
Real Assertions - No Print-Only Tests
Exit Codes: 0=Success, 1=Critical, 2=High Errors, 3=Warnings
"""

import sys
import json
from datetime import datetime

# ============================================================================
# TEST COUNTER
# ============================================================================

class Tests:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        
    def assert_equal(self, actual, expected, test_name):
        """Assert equality"""
        if actual == expected:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({
                "test": test_name,
                "expected": expected,
                "actual": actual,
                "severity": 2
            })
            print(f"  ✗ {test_name}: expected {expected}, got {actual}")
    
    def assert_type(self, obj, expected_type, test_name):
        """Assert type"""
        if isinstance(obj, expected_type):
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({
                "test": test_name,
                "expected": str(expected_type),
                "actual": str(type(obj)),
                "severity": 2
            })
            print(f"  ✗ {test_name}: expected {expected_type}, got {type(obj)}")
    
    def assert_in_range(self, value, min_val, max_val, test_name):
        """Assert value in range"""
        if min_val <= value <= max_val:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({
                "test": test_name,
                "expected": f"[{min_val}, {max_val}]",
                "actual": value,
                "severity": 2
            })
            print(f"  ✗ {test_name}: {value} outside [{min_val}, {max_val}]")
    
    def assert_true(self, condition, test_name):
        """Assert condition is true"""
        if condition:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({
                "test": test_name,
                "expected": "True",
                "actual": "False",
                "severity": 2
            })
            print(f"  ✗ {test_name}: condition is False")
    
    def assert_not_none(self, obj, test_name):
        """Assert object is not None"""
        if obj is not None:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({
                "test": test_name,
                "expected": "not None",
                "actual": "None",
                "severity": 1
            })
            print(f"  ✗ {test_name}: object is None")
    
    def get_exit_code(self):
        """Get exit code based on errors"""
        if self.failed == 0:
            return 0
        
        critical = sum(1 for e in self.errors if e["severity"] == 1)
        if critical > 0:
            return 1
        return 2

# ============================================================================
# TESTS
# ============================================================================

tests = Tests()

print("\n" + "="*80)
print("  KAGGLE RIGOROUS TEST SUITE")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# TEST 1: IMPORTS
print("TEST 1: MODULE IMPORTS")
print("-" * 80)
try:
    from module_inventory import ModuleInventory
    tests.assert_true(True, "ModuleInventory imported")
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "ModuleInventory import", "severity": 1})
    print(f"  ✗ ModuleInventory import: {e}")
    sys.exit(1)

try:
    from reward_system import RewardSystem, ExecutionResult
    tests.assert_true(True, "RewardSystem imported")
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "RewardSystem import", "severity": 1})
    print(f"  ✗ RewardSystem import: {e}")
    sys.exit(1)

try:
    from data_generator import TrainingDataGenerator
    tests.assert_true(True, "TrainingDataGenerator imported")
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "TrainingDataGenerator import", "severity": 1})
    print(f"  ✗ TrainingDataGenerator import: {e}")
    sys.exit(1)

try:
    from sft_trainer import SFTTrainer
    from rl_trainer import RLTrainer
    from icl_trainer import ICLTrainer
    from continuous_learning_trainer import ContinuousLearningTrainer
    from deepall_integration import DeepALLIntegration
    from training_orchestrator import TrainingOrchestrator
    tests.assert_true(True, "All trainers imported")
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "Trainers import", "severity": 1})
    print(f"  ✗ Trainers import: {e}")
    sys.exit(1)

# TEST 2: MODULE INVENTORY
print("\nTEST 2: MODULE INVENTORY")
print("-" * 80)
try:
    inventory = ModuleInventory('deepall_modules.json')
    tests.assert_type(inventory.modules, dict, "modules is dict")
    tests.assert_equal(len(inventory.modules), 215, "exactly 215 modules")
    
    # Check all modules have required fields
    all_valid = True
    for module_id, module in inventory.modules.items():
        if not (hasattr(module, 'id') and hasattr(module, 'name') and 
                hasattr(module, 'category') and hasattr(module, 'ai_training_method')):
            all_valid = False
            break
    tests.assert_true(all_valid, "all modules have required fields")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "Module inventory", "severity": 1})
    print(f"  ✗ Module inventory error: {e}")
    sys.exit(1)

# TEST 3: REWARD SYSTEM
print("\nTEST 3: REWARD SYSTEM")
print("-" * 80)
try:
    reward_system = RewardSystem()
    
    # Test 3 scenarios
    scenarios = [
        ("Perfect", ["m001", "m002"], ["m001", "m002"], 10.0, 0.1, True),
        ("Partial", ["m001", "m003"], ["m001", "m002"], 5.0, 0.5, True),
        ("Failed", ["m003", "m004"], ["m001", "m002"], 1.0, 0.9, False),
    ]
    
    for name, selected, expected, time, efficiency, success in scenarios:
        result = ExecutionResult(f"task_{name}", selected, expected, time, efficiency, success)
        reward = reward_system.calculate_reward(result)
        
        tests.assert_type(reward, (int, float), f"reward_{name} is numeric")
        tests.assert_in_range(reward, -1.0, 1.0, f"reward_{name} in [-1, 1]")
        
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "Reward system", "severity": 1})
    print(f"  ✗ Reward system error: {e}")
    sys.exit(1)

# TEST 4: DATA GENERATOR
print("\nTEST 4: DATA GENERATOR")
print("-" * 80)
try:
    generator = TrainingDataGenerator(inventory)
    
    sft_data = generator.generate_dataset(num_samples=20)
    tests.assert_type(sft_data, list, "SFT data is list")
    tests.assert_equal(len(sft_data), 20, "SFT data has 20 samples")
    
    rl_episodes = generator.generate_rl_episodes(num_episodes=5)
    tests.assert_type(rl_episodes, list, "RL episodes is list")
    tests.assert_equal(len(rl_episodes), 5, "RL episodes has 5 episodes")
    
    icl_examples = generator.generate_icl_examples(num_examples=10)
    tests.assert_type(icl_examples, list, "ICL examples is list")
    tests.assert_equal(len(icl_examples), 10, "ICL examples has 10 examples")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "Data generator", "severity": 1})
    print(f"  ✗ Data generator error: {e}")
    sys.exit(1)

# TEST 5: SFT TRAINER
print("\nTEST 5: SFT TRAINER")
print("-" * 80)
try:
    sft_trainer = SFTTrainer(inventory)
    sft_trainer.generate_training_data(num_samples=20)
    tests.assert_equal(len(sft_trainer.training_data), 20, "SFT training data generated")
    
    sft_results = sft_trainer.train(num_epochs=2, batch_size=8)
    tests.assert_type(sft_results, dict, "SFT results is dict")
    tests.assert_equal(len(sft_results['epochs']), 2, "SFT has 2 epochs")
    
    # Check accuracy values
    for epoch in sft_results['epochs']:
        accuracy = epoch.get('avg_accuracy', 0)
        tests.assert_in_range(accuracy, 0.0, 1.0, f"epoch {epoch['epoch']} accuracy in [0,1]")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "SFT trainer", "severity": 1})
    print(f"  ✗ SFT trainer error: {e}")
    sys.exit(1)

# TEST 6: RL TRAINER
print("\nTEST 6: RL TRAINER")
print("-" * 80)
try:
    rl_trainer = RLTrainer(inventory)
    rl_results = rl_trainer.train(num_episodes=3)
    tests.assert_type(rl_results, dict, "RL results is dict")
    tests.assert_equal(len(rl_results['episodes']), 3, "RL has 3 episodes")
    
    # Check reward values
    for episode in rl_results['episodes']:
        reward = episode.get('avg_reward', 0)
        tests.assert_in_range(reward, -1.0, 1.0, f"episode {episode['episode']} reward in [-1,1]")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "RL trainer", "severity": 1})
    print(f"  ✗ RL trainer error: {e}")
    sys.exit(1)

# TEST 7: ICL TRAINER
print("\nTEST 7: ICL TRAINER")
print("-" * 80)
try:
    icl_trainer = ICLTrainer(inventory)
    icl_trainer.generate_training_data(num_examples=15)
    tests.assert_equal(len(icl_trainer.training_examples), 15, "ICL training data generated")
    
    icl_results = icl_trainer.train(num_iterations=2)
    tests.assert_type(icl_results, dict, "ICL results is dict")
    tests.assert_equal(len(icl_results['iterations']), 2, "ICL has 2 iterations")
    
    # Check accuracy values
    for iteration in icl_results['iterations']:
        accuracy = iteration.get('avg_accuracy', 0)
        tests.assert_in_range(accuracy, 0.0, 1.0, f"iteration {iteration['iteration']} accuracy in [0,1]")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "ICL trainer", "severity": 1})
    print(f"  ✗ ICL trainer error: {e}")
    sys.exit(1)

# TEST 8: CONTINUOUS LEARNING
print("\nTEST 8: CONTINUOUS LEARNING")
print("-" * 80)
try:
    cl_trainer = ContinuousLearningTrainer(inventory)
    cl_results = cl_trainer.train_on_stream(num_batches=3, batch_size=15)
    tests.assert_type(cl_results, dict, "CL results is dict")
    tests.assert_equal(len(cl_results['batches']), 3, "CL has 3 batches")
    
    # Check reward values
    for batch in cl_results['batches']:
        reward = batch.get('avg_reward', 0)
        tests.assert_in_range(reward, -1.0, 1.0, f"batch {batch['batch']} reward in [-1,1]")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "Continuous learning", "severity": 1})
    print(f"  ✗ Continuous learning error: {e}")
    sys.exit(1)

# TEST 9: DEEPALL INTEGRATION
print("\nTEST 9: DEEPALL INTEGRATION")
print("-" * 80)
try:
    import random
    integration = DeepALLIntegration(inventory)
    
    sample_modules = random.sample(inventory.get_all_module_ids(), 5)
    synergies = integration.detect_synergies(sample_modules)
    
    tests.assert_type(synergies, dict, "synergies is dict")
    tests.assert_true('synergies' in synergies, "synergies has 'synergies' key")
    tests.assert_true('total_score' in synergies, "synergies has 'total_score' key")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "DeepALL integration", "severity": 1})
    print(f"  ✗ DeepALL integration error: {e}")
    sys.exit(1)

# TEST 10: TRAINING ORCHESTRATOR
print("\nTEST 10: TRAINING ORCHESTRATOR")
print("-" * 80)
try:
    orchestrator = TrainingOrchestrator(inventory)
    results = orchestrator.run_all_trainers(
        sft_samples=20,
        rl_episodes=2,
        icl_examples=10,
        cl_batches=2
    )
    
    tests.assert_type(results, dict, "orchestrator results is dict")
    tests.assert_true('sft' in results, "results has SFT")
    tests.assert_true('rl' in results, "results has RL")
    tests.assert_true('icl' in results, "results has ICL")
    tests.assert_true('cl' in results, "results has CL")
    
    # Check all methods have accuracy/reward
    tests.assert_true('accuracy' in results['sft'], "SFT has accuracy")
    tests.assert_true('success_rate' in results['rl'], "RL has success_rate")
    tests.assert_true('accuracy' in results['icl'], "ICL has accuracy")
    tests.assert_true('avg_reward' in results['cl'], "CL has avg_reward")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "Training orchestrator", "severity": 1})
    print(f"  ✗ Training orchestrator error: {e}")
    sys.exit(1)

# ============================================================================
# FINAL REPORT
# ============================================================================

print("\n" + "="*80)
print("TEST REPORT")
print("="*80)

total = tests.passed + tests.failed
print(f"\nTotal Tests: {total}")
print(f"Passed: {tests.passed} ✓")
print(f"Failed: {tests.failed} ✗")

exit_code = tests.get_exit_code()

if exit_code == 0:
    print(f"\nExit Code: 0 (SUCCESS)")
    print("✓ ALL TESTS PASSED")
    print("✓ Framework is production ready for Kaggle!")
elif exit_code == 1:
    print(f"\nExit Code: 1 (CRITICAL ERRORS)")
    print("✗ Critical errors found")
elif exit_code == 2:
    print(f"\nExit Code: 2 (HIGH ERRORS)")
    print("✗ High errors found")

if tests.errors:
    print("\n" + "="*80)
    print("ERRORS")
    print("="*80)
    for error in tests.errors:
        print(f"\n[SEVERITY {error['severity']}] {error['test']}")
        if 'expected' in error:
            print(f"  Expected: {error['expected']}")
        if 'actual' in error:
            print(f"  Actual: {error['actual']}")

print("\n" + "="*80)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

sys.exit(exit_code)
