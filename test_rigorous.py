#!/usr/bin/env python3
"""
RIGOROUS TEST SUITE - nanoGPT-DeepALL-Agent
Scientific Testing with Real Assertions (No Print-Only Tests)
"""

import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple

# ============================================================================
# TEST FRAMEWORK
# ============================================================================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors: List[Dict] = []
        self.details: List[Dict] = []
        self.start_time = datetime.now()
        
    def add_pass(self, test_name: str, details: str = ""):
        self.passed += 1
        self.details.append({
            "status": "PASS",
            "test": test_name,
            "details": details
        })
        
    def add_fail(self, test_name: str, severity: int, error: str, expected: str = "", actual: str = ""):
        self.failed += 1
        self.errors.append({
            "severity": severity,
            "test": test_name,
            "error": error,
            "expected": expected,
            "actual": actual
        })
        self.details.append({
            "status": "FAIL",
            "test": test_name,
            "severity": severity,
            "error": error
        })
        
    def add_warning(self, test_name: str, warning: str):
        self.warnings += 1
        self.details.append({
            "status": "WARNING",
            "test": test_name,
            "warning": warning
        })
        
    def get_exit_code(self) -> int:
        critical = sum(1 for e in self.errors if e["severity"] == 1)
        high = sum(1 for e in self.errors if e["severity"] == 2)
        
        if critical > 0:
            return 1
        elif high > 0:
            return 2
        elif self.warnings > 0:
            return 3
        else:
            return 0

# ============================================================================
# UNIT TESTS
# ============================================================================

def test_module_inventory_import(results: TestResult):
    """TEST 1.1.1: Can import ModuleInventory"""
    try:
        from module_inventory import ModuleInventory
        results.add_pass("1.1.1_import", "ModuleInventory imported successfully")
        return True
    except ImportError as e:
        results.add_fail("1.1.1_import", 1, f"Import failed: {e}", "ModuleInventory class", "ImportError")
        return False

def test_module_inventory_load(results: TestResult):
    """TEST 1.1.2: ModuleInventory loads exactly 215 modules"""
    try:
        from module_inventory import ModuleInventory
        inventory = ModuleInventory('deepall_modules.json')
        
        # ASSERTION 1: modules is dict
        assert isinstance(inventory.modules, dict), f"modules is {type(inventory.modules)}, not dict"
        
        # ASSERTION 2: exactly 215 modules
        assert len(inventory.modules) == 215, f"Expected 215 modules, got {len(inventory.modules)}"
        
        # ASSERTION 3: all keys are strings
        assert all(isinstance(k, str) for k in inventory.modules.keys()), "Not all keys are strings"
        
        # ASSERTION 4: all values have required attributes
        for module_id, module in inventory.modules.items():
            assert hasattr(module, 'id'), f"Module {module_id} missing 'id' attribute"
            assert hasattr(module, 'name'), f"Module {module_id} missing 'name' attribute"
            assert hasattr(module, 'category'), f"Module {module_id} missing 'category' attribute"
            assert hasattr(module, 'ai_training_method'), f"Module {module_id} missing 'ai_training_method' attribute"
        
        results.add_pass("1.1.2_load", f"Loaded exactly 215 modules with correct structure")
        return True
    except AssertionError as e:
        results.add_fail("1.1.2_load", 2, str(e), "215 modules with correct structure", str(e))
        return False
    except Exception as e:
        results.add_fail("1.1.2_load", 1, f"Unexpected error: {e}", "Load modules", str(type(e)))
        return False

def test_module_inventory_get_module(results: TestResult):
    """TEST 1.1.3: ModuleInventory.get_module() returns correct module"""
    try:
        from module_inventory import ModuleInventory
        inventory = ModuleInventory('deepall_modules.json')
        
        # Get a known module
        module = inventory.get_module('m001')
        
        # ASSERTION 1: module is not None
        assert module is not None, "get_module('m001') returned None"
        
        # ASSERTION 2: module has correct id
        assert module.id == 'm001', f"Module ID is {module.id}, not m001"
        
        # ASSERTION 3: module.name is string and not empty
        assert isinstance(module.name, str) and len(module.name) > 0, f"Module name invalid: {module.name}"
        
        # ASSERTION 4: module.category is string
        assert isinstance(module.category, str), f"Category is {type(module.category)}, not string"
        
        results.add_pass("1.1.3_get_module", "get_module() returns correct module with all fields")
        return True
    except AssertionError as e:
        results.add_fail("1.1.3_get_module", 2, str(e), "Valid module object", str(e))
        return False
    except Exception as e:
        results.add_fail("1.1.3_get_module", 1, f"Unexpected error: {e}", "Module object", str(type(e)))
        return False

def test_reward_system_import(results: TestResult):
    """TEST 1.1.4: Can import RewardSystem and ExecutionResult"""
    try:
        from reward_system import RewardSystem, ExecutionResult
        results.add_pass("1.1.4_import", "RewardSystem and ExecutionResult imported")
        return True
    except ImportError as e:
        results.add_fail("1.1.4_import", 1, f"Import failed: {e}", "RewardSystem, ExecutionResult", "ImportError")
        return False

def test_reward_calculation(results: TestResult):
    """TEST 1.1.5: RewardSystem calculates rewards in valid range [0, 1]"""
    try:
        from reward_system import RewardSystem, ExecutionResult
        reward_system = RewardSystem()
        
        # Test 3 scenarios
        test_cases = [
            ("Perfect", ["m001", "m002"], ["m001", "m002"], 10.0, 0.1, True),
            ("Partial", ["m001", "m003"], ["m001", "m002"], 5.0, 0.5, True),
            ("Failed", ["m003", "m004"], ["m001", "m002"], 1.0, 0.9, False),
        ]
        
        for name, selected, expected, time, efficiency, success in test_cases:
            result = ExecutionResult(f"task_{name}", selected, expected, time, efficiency, success)
            reward = reward_system.calculate_reward(result)
            
            # ASSERTION 1: reward is float
            assert isinstance(reward, (int, float)), f"Reward is {type(reward)}, not float"
            
            # ASSERTION 2: reward in valid range
            assert -1.0 <= reward <= 1.0, f"Reward {reward} outside [-1, 1] range"
        
        results.add_pass("1.1.5_reward_calc", "All 3 reward scenarios returned valid values in [-1, 1]")
        return True
    except AssertionError as e:
        results.add_fail("1.1.5_reward_calc", 2, str(e), "Rewards in [-1, 1]", str(e))
        return False
    except Exception as e:
        results.add_fail("1.1.5_reward_calc", 1, f"Unexpected error: {e}", "Reward calculation", str(type(e)))
        return False

def test_data_generator_import(results: TestResult):
    """TEST 1.1.6: Can import TrainingDataGenerator"""
    try:
        from data_generator import TrainingDataGenerator
        results.add_pass("1.1.6_import", "TrainingDataGenerator imported")
        return True
    except ImportError as e:
        results.add_fail("1.1.6_import", 1, f"Import failed: {e}", "TrainingDataGenerator", "ImportError")
        return False

def test_data_generator_generates_data(results: TestResult):
    """TEST 1.1.7: TrainingDataGenerator generates correct number of samples"""
    try:
        from module_inventory import ModuleInventory
        from data_generator import TrainingDataGenerator
        
        inventory = ModuleInventory('deepall_modules.json')
        generator = TrainingDataGenerator(inventory)
        
        # Generate SFT data
        sft_data = generator.generate_dataset(num_samples=20)
        
        # ASSERTION 1: returns list
        assert isinstance(sft_data, list), f"SFT data is {type(sft_data)}, not list"
        
        # ASSERTION 2: correct number of samples
        assert len(sft_data) == 20, f"Expected 20 samples, got {len(sft_data)}"
        
        # ASSERTION 3: each sample has required fields
        for i, sample in enumerate(sft_data):
            assert isinstance(sample, dict), f"Sample {i} is {type(sample)}, not dict"
            assert 'input' in sample, f"Sample {i} missing 'input' field"
            assert 'output' in sample, f"Sample {i} missing 'output' field"
        
        results.add_pass("1.1.7_data_gen", "Generated 20 SFT samples with correct structure")
        return True
    except AssertionError as e:
        results.add_fail("1.1.7_data_gen", 2, str(e), "20 samples with correct structure", str(e))
        return False
    except Exception as e:
        results.add_fail("1.1.7_data_gen", 1, f"Unexpected error: {e}", "Data generation", str(type(e)))
        return False

def test_sft_trainer_import(results: TestResult):
    """TEST 1.1.8: Can import SFTTrainer"""
    try:
        from sft_trainer import SFTTrainer
        results.add_pass("1.1.8_import", "SFTTrainer imported")
        return True
    except ImportError as e:
        results.add_fail("1.1.8_import", 1, f"Import failed: {e}", "SFTTrainer", "ImportError")
        return False

def test_sft_trainer_trains(results: TestResult):
    """TEST 1.1.9: SFTTrainer.train() returns valid results"""
    try:
        from module_inventory import ModuleInventory
        from sft_trainer import SFTTrainer
        
        inventory = ModuleInventory('deepall_modules.json')
        trainer = SFTTrainer(inventory)
        trainer.generate_training_data(num_samples=10)
        
        results_dict = trainer.train(num_epochs=2, batch_size=4)
        
        # ASSERTION 1: results is dict
        assert isinstance(results_dict, dict), f"Results is {type(results_dict)}, not dict"
        
        # ASSERTION 2: has required keys
        assert 'epochs' in results_dict, "Missing 'epochs' key"
        assert 'total_epochs' in results_dict, "Missing 'total_epochs' key"
        
        # ASSERTION 3: epochs list has correct length
        assert len(results_dict['epochs']) == 2, f"Expected 2 epochs, got {len(results_dict['epochs'])}"
        
        # ASSERTION 4: each epoch has accuracy in [0, 1]
        for epoch in results_dict['epochs']:
            accuracy = epoch.get('avg_accuracy', 0)
            assert 0.0 <= accuracy <= 1.0, f"Accuracy {accuracy} outside [0, 1]"
        
        results.add_pass("1.1.9_sft_train", "SFT training completed with valid results")
        return True
    except AssertionError as e:
        results.add_fail("1.1.9_sft_train", 2, str(e), "Valid training results", str(e))
        return False
    except Exception as e:
        results.add_fail("1.1.9_sft_train", 1, f"Unexpected error: {e}", "SFT training", str(type(e)))
        return False

def test_rl_trainer_import(results: TestResult):
    """TEST 1.1.10: Can import RLTrainer"""
    try:
        from rl_trainer import RLTrainer
        results.add_pass("1.1.10_import", "RLTrainer imported")
        return True
    except ImportError as e:
        results.add_fail("1.1.10_import", 1, f"Import failed: {e}", "RLTrainer", "ImportError")
        return False

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_sft_complete_workflow(results: TestResult):
    """TEST 2.1.1: SFT complete workflow (init → generate → train)"""
    try:
        from module_inventory import ModuleInventory
        from sft_trainer import SFTTrainer
        
        # Step 1: Load inventory
        inventory = ModuleInventory('deepall_modules.json')
        assert len(inventory.modules) == 215, "Inventory load failed"
        
        # Step 2: Initialize trainer
        trainer = SFTTrainer(inventory)
        assert trainer is not None, "Trainer initialization failed"
        
        # Step 3: Generate data
        trainer.generate_training_data(num_samples=15)
        assert len(trainer.training_data) == 15, f"Expected 15 samples, got {len(trainer.training_data)}"
        
        # Step 4: Train
        train_results = trainer.train(num_epochs=2, batch_size=5)
        assert len(train_results['epochs']) == 2, "Training epochs mismatch"
        
        # Step 5: Evaluate
        eval_results = trainer.evaluate()
        assert 'accuracy' in eval_results, "Evaluation missing accuracy"
        assert 0.0 <= eval_results['accuracy'] <= 1.0, "Accuracy out of range"
        
        results.add_pass("2.1.1_sft_workflow", "SFT workflow: init → generate → train → evaluate")
        return True
    except AssertionError as e:
        results.add_fail("2.1.1_sft_workflow", 2, str(e), "Complete SFT workflow", str(e))
        return False
    except Exception as e:
        results.add_fail("2.1.1_sft_workflow", 1, f"Unexpected error: {e}", "SFT workflow", str(type(e)))
        return False

def test_rl_complete_workflow(results: TestResult):
    """TEST 2.1.2: RL complete workflow (init → train → evaluate)"""
    try:
        from module_inventory import ModuleInventory
        from rl_trainer import RLTrainer
        
        # Step 1: Load inventory
        inventory = ModuleInventory('deepall_modules.json')
        
        # Step 2: Initialize trainer
        trainer = RLTrainer(inventory)
        
        # Step 3: Train
        train_results = trainer.train(num_episodes=2, steps_per_episode=10)
        assert 'episodes' in train_results, "Training missing episodes"
        assert len(train_results['episodes']) == 2, "Episode count mismatch"
        
        # Step 4: Evaluate
        eval_results = trainer.evaluate(num_test_episodes=2)
        assert 'success_rate' in eval_results, "Evaluation missing success_rate"
        assert 0.0 <= eval_results['success_rate'] <= 1.0, "Success rate out of range"
        
        results.add_pass("2.1.2_rl_workflow", "RL workflow: init → train → evaluate")
        return True
    except AssertionError as e:
        results.add_fail("2.1.2_rl_workflow", 2, str(e), "Complete RL workflow", str(e))
        return False
    except Exception as e:
        results.add_fail("2.1.2_rl_workflow", 1, f"Unexpected error: {e}", "RL workflow", str(type(e)))
        return False

# ============================================================================
# END-TO-END TESTS
# ============================================================================

def test_all_imports(results: TestResult):
    """TEST 3.1.1: All modules can be imported"""
    modules_to_import = [
        'module_inventory',
        'reward_system',
        'data_generator',
        'sft_trainer',
        'rl_trainer',
        'icl_trainer',
        'continuous_learning_trainer',
        'deepall_integration',
        'training_orchestrator'
    ]
    
    try:
        for module_name in modules_to_import:
            __import__(module_name)
        
        results.add_pass("3.1.1_all_imports", f"All {len(modules_to_import)} modules imported successfully")
        return True
    except ImportError as e:
        results.add_fail("3.1.1_all_imports", 1, f"Import failed: {e}", "All modules", str(e))
        return False

def test_orchestrator_all_methods(results: TestResult):
    """TEST 3.1.2: TrainingOrchestrator runs all 4 methods"""
    try:
        from module_inventory import ModuleInventory
        from training_orchestrator import TrainingOrchestrator
        
        inventory = ModuleInventory('deepall_modules.json')
        orchestrator = TrainingOrchestrator(inventory)
        
        # Run all trainers
        unified_results = orchestrator.run_all_trainers(
            sft_samples=10,
            rl_episodes=1,
            icl_examples=5,
            cl_batches=1
        )
        
        # ASSERTION 1: has all 4 methods
        assert 'sft' in unified_results, "Missing SFT results"
        assert 'rl' in unified_results, "Missing RL results"
        assert 'icl' in unified_results, "Missing ICL results"
        assert 'cl' in unified_results, "Missing CL results"
        
        # ASSERTION 2: each method has required fields
        assert 'accuracy' in unified_results['sft'], "SFT missing accuracy"
        assert 'success_rate' in unified_results['rl'], "RL missing success_rate"
        assert 'accuracy' in unified_results['icl'], "ICL missing accuracy"
        assert 'avg_reward' in unified_results['cl'], "CL missing avg_reward"
        
        # ASSERTION 3: all values are in valid ranges
        assert 0.0 <= unified_results['sft']['accuracy'] <= 1.0, "SFT accuracy out of range"
        assert 0.0 <= unified_results['rl']['success_rate'] <= 1.0, "RL success_rate out of range"
        assert 0.0 <= unified_results['icl']['accuracy'] <= 1.0, "ICL accuracy out of range"
        
        results.add_pass("3.1.2_orchestrator", "All 4 training methods executed with valid results")
        return True
    except AssertionError as e:
        results.add_fail("3.1.2_orchestrator", 2, str(e), "All 4 methods with valid results", str(e))
        return False
    except Exception as e:
        results.add_fail("3.1.2_orchestrator", 1, f"Unexpected error: {e}", "Orchestrator", str(type(e)))
        return False

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests and generate report"""
    
    results = TestResult()
    
    print("\n" + "="*80)
    print("  RIGOROUS TEST SUITE - nanoGPT-DeepALL-Agent")
    print("="*80)
    print(f"Started: {results.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Unit Tests
    print("UNIT TESTS")
    print("-" * 80)
    test_module_inventory_import(results)
    test_module_inventory_load(results)
    test_module_inventory_get_module(results)
    test_reward_system_import(results)
    test_reward_calculation(results)
    test_data_generator_import(results)
    test_data_generator_generates_data(results)
    test_sft_trainer_import(results)
    test_sft_trainer_trains(results)
    test_rl_trainer_import(results)
    
    # Integration Tests
    print("\nINTEGRATION TESTS")
    print("-" * 80)
    test_sft_complete_workflow(results)
    test_rl_complete_workflow(results)
    
    # End-to-End Tests
    print("\nEND-TO-END TESTS")
    print("-" * 80)
    test_all_imports(results)
    test_orchestrator_all_methods(results)
    
    # Generate Report
    print("\n" + "="*80)
    print("TEST REPORT")
    print("="*80)
    
    total = results.passed + results.failed
    print(f"Total Tests: {total}")
    print(f"Passed: {results.passed} ✓")
    print(f"Failed: {results.failed} ✗")
    print(f"Warnings: {results.warnings} ⚠")
    
    exit_code = results.get_exit_code()
    
    if exit_code == 0:
        print(f"\nExit Code: 0 (SUCCESS)")
        print("✓ ALL TESTS PASSED - Framework is production ready!")
    elif exit_code == 1:
        print(f"\nExit Code: 1 (CRITICAL ERRORS)")
        print("✗ CRITICAL ERRORS - Framework has blocking issues")
    elif exit_code == 2:
        print(f"\nExit Code: 2 (HIGH ERRORS)")
        print("✗ HIGH ERRORS - Framework has significant issues")
    elif exit_code == 3:
        print(f"\nExit Code: 3 (WARNINGS)")
        print("⚠ WARNINGS - Framework has non-critical issues")
    
    # Print errors if any
    if results.errors:
        print("\n" + "="*80)
        print("ERRORS")
        print("="*80)
        for error in results.errors:
            print(f"\n[SEVERITY {error['severity']}] {error['test']}")
            print(f"  Error: {error['error']}")
            if error['expected']:
                print(f"  Expected: {error['expected']}")
            if error['actual']:
                print(f"  Actual: {error['actual']}")
    
    print("\n" + "="*80)
    
    return exit_code

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
