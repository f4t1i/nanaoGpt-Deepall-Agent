#!/usr/bin/env python3
"""
Integration Tests for Resilient-Nano-Trainer with nanoGPT-DeepALL-Agent
Tests ARS Optimizer integration with DeepALL modules and training
"""

import sys
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Import core components
from module_inventory import ModuleInventory
from enhanced_module_inventory import EnhancedModuleInventory
from deepall_integration_extended import DeepALLIntegrationExtended
from reward_system import RewardSystem, ExecutionResult
from resilient_nano_integration import (
    ResilientNanoTrainingConfig,
    ResilientNanoTrainer
)


class TestCounter:
    """Simple test counter for tracking results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def assert_true(self, condition: bool, test_name: str):
        if condition:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append(test_name)
            print(f"  ✗ {test_name}")
    
    def assert_equal(self, actual, expected, test_name: str):
        if actual == expected:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append(f"{test_name}: expected {expected}, got {actual}")
            print(f"  ✗ {test_name}: expected {expected}, got {actual}")
    
    def assert_in_range(self, value: float, min_val: float, max_val: float, test_name: str):
        if min_val <= value <= max_val:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append(f"{test_name}: {value} not in [{min_val}, {max_val}]")
            print(f"  ✗ {test_name}: {value} not in [{min_val}, {max_val}]")
    
    def assert_not_none(self, obj, test_name: str):
        if obj is not None:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append(f"{test_name}: object is None")
            print(f"  ✗ {test_name}: object is None")
    
    def assert_greater_than(self, actual, threshold, test_name: str):
        if actual > threshold:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append(f"{test_name}: {actual} not > {threshold}")
            print(f"  ✗ {test_name}: {actual} not > {threshold}")


# ============================================================================
# CATEGORY 1: COMPONENT INITIALIZATION TESTS
# ============================================================================

def test_category_1_initialization(tests: TestCounter):
    """Test 1: Component Initialization"""
    print("\n" + "="*80)
    print("CATEGORY 1: COMPONENT INITIALIZATION TESTS")
    print("="*80)
    
    # Test 1.1: Module Inventory
    print("\n[1.1] Testing Module Inventory...")
    try:
        inventory = ModuleInventory('deepall_modules.json')
        tests.assert_equal(len(inventory.modules), 215, "Module count is 215")
        tests.assert_not_none(inventory.get_module('m001'), "Can get module m001")
    except Exception as e:
        tests.assert_true(False, f"Module Inventory initialization: {str(e)}")
    
    # Test 1.2: Enhanced Inventory
    print("\n[1.2] Testing Enhanced Inventory...")
    try:
        enhanced_inventory = EnhancedModuleInventory(
            'deepall_modules.json',
            'DeepALL_MASTER_V7_FINAL_WITH_ALL_REITERS.xlsx'
        )
        tests.assert_equal(len(enhanced_inventory.modules), 215, "Enhanced inventory has 215 modules")
        tests.assert_greater_than(len(enhanced_inventory.superintelligences_data), 0, "Has superintelligences")
    except Exception as e:
        tests.assert_true(False, f"Enhanced Inventory initialization: {str(e)}")
    
    # Test 1.3: DeepALL Integration
    print("\n[1.3] Testing DeepALL Integration...")
    try:
        integration = DeepALLIntegrationExtended(enhanced_inventory)
        tests.assert_not_none(integration, "DeepALL Integration initialized")
        tests.assert_greater_than(len(enhanced_inventory.superintelligences_data), 0, "Has superintelligences")
    except Exception as e:
        tests.assert_true(False, f"DeepALL Integration initialization: {str(e)}")
    
    # Test 1.4: Reward System
    print("\n[1.4] Testing Reward System...")
    try:
        reward_system = RewardSystem()
        result = ExecutionResult("test", ["m001"], ["m001"], 2.0, 0.5, True)
        reward = reward_system.calculate_reward(result)
        tests.assert_in_range(reward, -1.0, 1.0, "Reward in valid range [-1, 1]")
    except Exception as e:
        tests.assert_true(False, f"Reward System initialization: {str(e)}")
    
    # Test 1.5: Resilient Nano Config
    print("\n[1.5] Testing Resilient Nano Config...")
    try:
        config = ResilientNanoTrainingConfig()
        tests.assert_equal(config.ars_alpha, 2.0, "ARS alpha is 2.0")
        tests.assert_equal(config.num_epochs, 10, "Num epochs is 10")
        tests.assert_equal(config.num_modules, 5, "Num modules is 5")
    except Exception as e:
        tests.assert_true(False, f"Config initialization: {str(e)}")


# ============================================================================
# CATEGORY 2: MODULE SELECTION TESTS
# ============================================================================

def test_category_2_module_selection(tests: TestCounter, integration: DeepALLIntegrationExtended):
    """Test 2: Module Selection Strategies"""
    print("\n" + "="*80)
    print("CATEGORY 2: MODULE SELECTION TESTS")
    print("="*80)
    
    # Test 2.1: Basic Selection
    print("\n[2.1] Testing Basic Module Selection...")
    try:
        modules = integration.optimize_module_selection(5)
        tests.assert_equal(len(modules), 5, "Selected 5 modules")
        tests.assert_true(all(isinstance(m, str) for m in modules), "All modules are strings")
    except Exception as e:
        tests.assert_true(False, f"Basic selection: {str(e)}")
    
    # Test 2.2: SI-Aware Selection
    print("\n[2.2] Testing SI-Aware Selection...")
    try:
        modules = integration.optimize_by_superintelligence(3)
        tests.assert_equal(len(modules), 3, "Selected 3 modules")
        tests.assert_true(all(isinstance(m, str) for m in modules), "All modules are strings")
    except Exception as e:
        tests.assert_true(False, f"SI-aware selection: {str(e)}")
    
    # Test 2.3: Learning-Aware Selection
    print("\n[2.3] Testing Learning-Aware Selection...")
    try:
        modules = integration.optimize_for_learning(4)
        tests.assert_equal(len(modules), 4, "Selected 4 modules")
        tests.assert_true(all(isinstance(m, str) for m in modules), "All modules are strings")
    except Exception as e:
        tests.assert_true(False, f"Learning-aware selection: {str(e)}")
    
    # Test 2.4: Performance-Aware Selection
    print("\n[2.4] Testing Performance-Aware Selection...")
    try:
        modules = integration.optimize_for_performance(4)
        tests.assert_equal(len(modules), 4, "Selected 4 modules")
        tests.assert_true(all(isinstance(m, str) for m in modules), "All modules are strings")
    except Exception as e:
        tests.assert_true(False, f"Performance-aware selection: {str(e)}")
    
    # Test 2.5: Different Sizes
    print("\n[2.5] Testing Different Selection Sizes...")
    try:
        for size in [1, 3, 5, 10]:
            modules = integration.optimize_module_selection(size)
            tests.assert_equal(len(modules), size, f"Selected {size} modules")
    except Exception as e:
        tests.assert_true(False, f"Different sizes: {str(e)}")


# ============================================================================
# CATEGORY 3: SYNERGY DETECTION TESTS
# ============================================================================

def test_category_3_synergy_detection(tests: TestCounter, integration: DeepALLIntegrationExtended):
    """Test 3: Synergy Detection and Analysis"""
    print("\n" + "="*80)
    print("CATEGORY 3: SYNERGY DETECTION TESTS")
    print("="*80)
    
    # Test 3.1: Basic Synergy Detection
    print("\n[3.1] Testing Basic Synergy Detection...")
    try:
        modules = ['m001', 'm002', 'm003']
        synergies = integration.detect_synergies(modules)
        tests.assert_not_none(synergies, "Synergies detected")
        tests.assert_in_range(synergies['total_score'], 0.0, 10.0, "Score in valid range")
    except Exception as e:
        tests.assert_true(False, f"Basic synergy: {str(e)}")
    
    # Test 3.2: Synergy with Optimal Modules
    print("\n[3.2] Testing Synergy with Optimal Modules...")
    try:
        optimal = integration.optimize_module_selection(5)
        synergies = integration.detect_synergies(optimal)
        tests.assert_greater_than(synergies['total_score'], 0.0, "Optimal modules have positive synergy")
    except Exception as e:
        tests.assert_true(False, f"Optimal synergy: {str(e)}")
    
    # Test 3.3: Synergy Consistency
    print("\n[3.3] Testing Synergy Consistency...")
    try:
        modules = ['m001', 'm002', 'm003']
        score1 = integration.detect_synergies(modules)['total_score']
        score2 = integration.detect_synergies(modules)['total_score']
        tests.assert_equal(score1, score2, "Synergy scores are consistent")
    except Exception as e:
        tests.assert_true(False, f"Consistency: {str(e)}")
    
    # Test 3.4: Conflict Detection
    print("\n[3.4] Testing Conflict Detection...")
    try:
        modules = ['m001', 'm002', 'm003']
        conflicts = integration.detect_conflicts(modules)
        tests.assert_not_none(conflicts, "Conflicts detected")
        tests.assert_in_range(conflicts['conflict_score'], 0.0, 1.0, "Conflict score in [0, 1]")
    except Exception as e:
        tests.assert_true(False, f"Conflict detection: {str(e)}")


# ============================================================================
# CATEGORY 4: RESILIENT NANO TRAINER TESTS
# ============================================================================

def test_category_4_resilient_trainer(tests: TestCounter, enhanced_inventory, integration):
    """Test 4: Resilient Nano Trainer Integration"""
    print("\n" + "="*80)
    print("CATEGORY 4: RESILIENT NANO TRAINER TESTS")
    print("="*80)
    
    # Test 4.1: Trainer Initialization
    print("\n[4.1] Testing Trainer Initialization...")
    try:
        config = ResilientNanoTrainingConfig()
        trainer = ResilientNanoTrainer(enhanced_inventory, integration, config)
        tests.assert_not_none(trainer, "Trainer initialized")
        tests.assert_equal(trainer.config.num_modules, 5, "Config has 5 modules")
    except Exception as e:
        tests.assert_true(False, f"Trainer initialization: {str(e)}")
    
    # Test 4.2: Module Selection via Trainer
    print("\n[4.2] Testing Module Selection via Trainer...")
    try:
        config = ResilientNanoTrainingConfig()
        trainer = ResilientNanoTrainer(enhanced_inventory, integration, config)
        modules = trainer.select_optimal_modules(5)
        tests.assert_equal(len(modules), 5, "Selected 5 modules")
    except Exception as e:
        tests.assert_true(False, f"Module selection: {str(e)}")
    
    # Test 4.3: Configuration Variations
    print("\n[4.3] Testing Configuration Variations...")
    try:
        config1 = ResilientNanoTrainingConfig()
        config1.ars_alpha = 1.0
        trainer1 = ResilientNanoTrainer(enhanced_inventory, integration, config1)
        tests.assert_equal(trainer1.config.ars_alpha, 1.0, "Config1 alpha is 1.0")
        
        config2 = ResilientNanoTrainingConfig()
        config2.ars_alpha = 3.0
        trainer2 = ResilientNanoTrainer(enhanced_inventory, integration, config2)
        tests.assert_equal(trainer2.config.ars_alpha, 3.0, "Config2 alpha is 3.0")
    except Exception as e:
        tests.assert_true(False, f"Configuration variations: {str(e)}")
    
    # Test 4.4: Resilience Metrics Initialization
    print("\n[4.4] Testing Resilience Metrics...")
    try:
        config = ResilientNanoTrainingConfig()
        trainer = ResilientNanoTrainer(enhanced_inventory, integration, config)
        tests.assert_equal(trainer.resilience_metrics['total_steps'], 0, "Initial steps is 0")
        tests.assert_equal(trainer.resilience_metrics['recovery_events'], 0, "Initial recovery is 0")
    except Exception as e:
        tests.assert_true(False, f"Resilience metrics: {str(e)}")


# ============================================================================
# CATEGORY 5: TRAINING REPORT TESTS
# ============================================================================

def test_category_5_training_reports(tests: TestCounter, enhanced_inventory, integration):
    """Test 5: Training Reports and Logging"""
    print("\n" + "="*80)
    print("CATEGORY 5: TRAINING REPORT TESTS")
    print("="*80)
    
    # Test 5.1: Report Generation
    print("\n[5.1] Testing Report Generation...")
    try:
        config = ResilientNanoTrainingConfig()
        trainer = ResilientNanoTrainer(enhanced_inventory, integration, config)
        report = trainer.get_training_report()
        tests.assert_not_none(report, "Report generated")
        tests.assert_not_none(report.get('timestamp'), "Report has timestamp")
    except Exception as e:
        tests.assert_true(False, f"Report generation: {str(e)}")
    
    # Test 5.2: Report Structure
    print("\n[5.2] Testing Report Structure...")
    try:
        config = ResilientNanoTrainingConfig()
        trainer = ResilientNanoTrainer(enhanced_inventory, integration, config)
        report = trainer.get_training_report()
        tests.assert_not_none(report.get('total_modules_trained'), "Has total_modules_trained")
        tests.assert_not_none(report.get('total_training_steps'), "Has total_training_steps")
        tests.assert_not_none(report.get('config'), "Has config")
    except Exception as e:
        tests.assert_true(False, f"Report structure: {str(e)}")
    
    # Test 5.3: Report Metrics
    print("\n[5.3] Testing Report Metrics...")
    try:
        config = ResilientNanoTrainingConfig()
        trainer = ResilientNanoTrainer(enhanced_inventory, integration, config)
        report = trainer.get_training_report()
        tests.assert_in_range(report['total_modules_trained'], 0, 215, "Modules in valid range")
        tests.assert_in_range(report['total_training_steps'], 0, 1000000, "Steps in valid range")
    except Exception as e:
        tests.assert_true(False, f"Report metrics: {str(e)}")


# ============================================================================
# CATEGORY 6: DATA INTEGRITY TESTS
# ============================================================================

def test_category_6_data_integrity(tests: TestCounter, enhanced_inventory, integration):
    """Test 6: Data Integrity and Consistency"""
    print("\n" + "="*80)
    print("CATEGORY 6: DATA INTEGRITY TESTS")
    print("="*80)
    
    # Test 6.1: Module Data Consistency
    print("\n[6.1] Testing Module Data Consistency...")
    try:
        module1 = enhanced_inventory.get_module('m001')
        module2 = enhanced_inventory.get_module('m001')
        tests.assert_equal(module1, module2, "Same module returns same data")
    except Exception as e:
        tests.assert_true(False, f"Data consistency: {str(e)}")
    
    # Test 6.2: Superintelligence Data
    print("\n[6.2] Testing Superintelligence Data...")
    try:
        si_info = enhanced_inventory.get_superintelligence_info('si001')
        tests.assert_not_none(si_info, "SI info retrieved")
    except Exception as e:
        tests.assert_true(False, f"SI data: {str(e)}")
    
    # Test 6.3: No Data Loss
    print("\n[6.3] Testing No Data Loss...")
    try:
        all_modules = enhanced_inventory.get_all_module_ids()
        tests.assert_equal(len(all_modules), 215, "All 215 modules present")
    except Exception as e:
        tests.assert_true(False, f"Data loss check: {str(e)}")


# ============================================================================
# CATEGORY 7: EDGE CASES AND ERROR HANDLING
# ============================================================================

def test_category_7_edge_cases(tests: TestCounter, integration):
    """Test 7: Edge Cases and Error Handling"""
    print("\n" + "="*80)
    print("CATEGORY 7: EDGE CASES AND ERROR HANDLING")
    print("="*80)
    
    # Test 7.1: Single Module Selection
    print("\n[7.1] Testing Single Module Selection...")
    try:
        modules = integration.optimize_module_selection(1)
        tests.assert_equal(len(modules), 1, "Selected 1 module")
    except Exception as e:
        tests.assert_true(False, f"Single module: {str(e)}")
    
    # Test 7.2: Large Selection
    print("\n[7.2] Testing Large Module Selection...")
    try:
        modules = integration.optimize_module_selection(50)
        tests.assert_equal(len(modules), 50, "Selected 50 modules")
    except Exception as e:
        tests.assert_true(False, f"Large selection: {str(e)}")
    
    # Test 7.3: All Modules
    print("\n[7.3] Testing All Modules Selection...")
    try:
        modules = integration.optimize_module_selection(215)
        tests.assert_equal(len(modules), 215, "Selected all 215 modules")
    except Exception as e:
        tests.assert_true(False, f"All modules: {str(e)}")
    
    # Test 7.4: Duplicate Handling
    print("\n[7.4] Testing Duplicate Handling...")
    try:
        modules = ['m001', 'm001', 'm002']
        synergies = integration.detect_synergies(modules)
        tests.assert_not_none(synergies, "Handles duplicates gracefully")
    except Exception as e:
        tests.assert_true(False, f"Duplicate handling: {str(e)}")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*80)
    print("RESILIENT-NANO-TRAINER INTEGRATION TESTS")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = TestCounter()
    
    # Initialize components
    print("\n[SETUP] Initializing components...")
    try:
        inventory = ModuleInventory('deepall_modules.json')
        enhanced_inventory = EnhancedModuleInventory(
            'deepall_modules.json',
            'DeepALL_MASTER_V7_FINAL_WITH_ALL_REITERS.xlsx'
        )
        integration = DeepALLIntegrationExtended(enhanced_inventory)
        print("  ✓ All components initialized")
    except Exception as e:
        print(f"  ✗ Setup failed: {str(e)}")
        return
    
    # Run test categories
    test_category_1_initialization(tests)
    test_category_2_module_selection(tests, integration)
    test_category_3_synergy_detection(tests, integration)
    test_category_4_resilient_trainer(tests, enhanced_inventory, integration)
    test_category_5_training_reports(tests, enhanced_inventory, integration)
    test_category_6_data_integrity(tests, enhanced_inventory, integration)
    test_category_7_edge_cases(tests, integration)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {tests.passed + tests.failed}")
    print(f"Passed: {tests.passed} ✓")
    print(f"Failed: {tests.failed} ✗")
    print(f"Success Rate: {100 * tests.passed / (tests.passed + tests.failed):.1f}%")
    
    if tests.failed > 0:
        print("\nFailed Tests:")
        for error in tests.errors:
            print(f"  - {error}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Return exit code
    return 0 if tests.failed == 0 else 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
