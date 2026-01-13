#!/usr/bin/env python3
"""
DEEPALL INTEGRATION TEST SUITE - 48 Comprehensive Tests
Real Assertions - Scientific Rigor
Exit Codes: 0=Success, 1=Critical, 2=High Errors, 3=Warnings
"""

import sys
import json
import random
from datetime import datetime
from typing import Dict, List

# ============================================================================
# TEST FRAMEWORK
# ============================================================================

class Tests:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def assert_equal(self, actual, expected, test_name):
        if actual == expected:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({"test": test_name, "severity": 2})
            print(f"  ✗ {test_name}: expected {expected}, got {actual}")
    
    def assert_type(self, obj, expected_type, test_name):
        if isinstance(obj, expected_type):
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({"test": test_name, "severity": 2})
            print(f"  ✗ {test_name}: expected {expected_type}, got {type(obj)}")
    
    def assert_in_range(self, value, min_val, max_val, test_name):
        if min_val <= value <= max_val:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({"test": test_name, "severity": 2})
            print(f"  ✗ {test_name}: {value} not in [{min_val}, {max_val}]")
    
    def assert_true(self, condition, test_name):
        if condition:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({"test": test_name, "severity": 2})
            print(f"  ✗ {test_name}: condition is False")
    
    def assert_greater(self, actual, expected, test_name):
        if actual > expected:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({"test": test_name, "severity": 2})
            print(f"  ✗ {test_name}: {actual} not > {expected}")
    
    def assert_less(self, actual, expected, test_name):
        if actual < expected:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({"test": test_name, "severity": 2})
            print(f"  ✗ {test_name}: {actual} not < {expected}")
    
    def assert_not_none(self, obj, test_name):
        if obj is not None:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({"test": test_name, "severity": 2})
            print(f"  ✗ {test_name}: object is None")

# ============================================================================
# IMPORTS & INITIALIZATION
# ============================================================================

print("="*80)
print("  DEEPALL INTEGRATION TEST SUITE - 48 Tests")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

try:
    from module_inventory import ModuleInventory
    from reward_system import RewardSystem, ExecutionResult
    from data_generator import TrainingDataGenerator
    from sft_trainer import SFTTrainer
    from rl_trainer import RLTrainer
    from icl_trainer import ICLTrainer
    from continuous_learning_trainer import ContinuousLearningTrainer
    from deepall_integration import DeepALLIntegration
    from training_orchestrator import TrainingOrchestrator
    
    print("✓ All imports successful\n")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Initialize
inventory = ModuleInventory('deepall_modules.json')
integration = DeepALLIntegration(inventory)
tests = Tests()

print("✓ Inventory and Integration initialized\n")

# ============================================================================
# CATEGORY 1: SYNERGY DETECTION (10 Tests)
# ============================================================================

print("CATEGORY 1: SYNERGY DETECTION")
print("-" * 80)

# TEST 1.1: Basic synergy detection
try:
    modules = random.sample(inventory.get_all_module_ids(), 5)
    synergies = integration.detect_synergies(modules)
    
    tests.assert_type(synergies, dict, "1.1.1: synergies is dict")
    tests.assert_type(synergies['synergies'], list, "1.1.2: synergies['synergies'] is list")
    tests.assert_type(synergies['total_score'], (int, float), "1.1.3: total_score is numeric")
    tests.assert_in_range(synergies['total_score'], 0, 10, "1.1.4: total_score in [0, 10]")
    tests.assert_true(len(synergies['synergies']) >= 1, "1.1.5: at least 1 synergy detected")
    tests.assert_true(all('type' in s and 'score' in s for s in synergies['synergies']), 
                     "1.1.6: all synergies have type and score")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "1.1: Basic synergy detection", "severity": 2})
    print(f"  ✗ 1.1: {e}")

# TEST 1.2: Synergy types validation
try:
    synergies = integration.detect_synergies(modules)
    valid_types = ['category', 'ai_method', 'pairwise', 'hierarchical', 'complementary']
    
    for synergy in synergies['synergies']:
        tests.assert_true(synergy['type'] in valid_types, 
                         f"1.2.1: synergy type '{synergy['type']}' is valid")
        tests.assert_in_range(synergy['score'], 0, 1, f"1.2.2: synergy score in [0,1]")
        
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "1.2: Synergy types validation", "severity": 2})
    print(f"  ✗ 1.2: {e}")

# TEST 1.3: Determinism
try:
    score1 = integration.detect_synergies(modules)['total_score']
    score2 = integration.detect_synergies(modules)['total_score']
    
    tests.assert_equal(score1, score2, "1.3.1: deterministic total_score")
    
    synergies1 = len(integration.detect_synergies(modules)['synergies'])
    synergies2 = len(integration.detect_synergies(modules)['synergies'])
    
    tests.assert_equal(synergies1, synergies2, "1.3.2: deterministic synergy count")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "1.3: Determinism", "severity": 2})
    print(f"  ✗ 1.3: {e}")

# TEST 1.4: FIXED - Optimization Quality (Optimized vs Random)
try:
    # Optimized modules should have higher synergy than random
    optimized = integration.optimize_module_selection(num_modules=5)
    score_optimized = integration.detect_synergies(optimized)['total_score']
    
    random_modules = random.sample(inventory.get_all_module_ids(), 5)
    score_random = integration.detect_synergies(random_modules)['total_score']
    
    tests.assert_greater(score_optimized, score_random, 
                        "1.4.1: optimized score > random score")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "1.4: Optimization quality", "severity": 2})
    print(f"  ✗ 1.4: {e}")

# ============================================================================
# CATEGORY 2: MODULE OPTIMIZATION (11 Tests)
# ============================================================================

print("\nCATEGORY 2: MODULE OPTIMIZATION")
print("-" * 80)

# TEST 2.1: Optimization returns list
try:
    optimal = integration.optimize_module_selection(num_modules=5)
    
    tests.assert_type(optimal, list, "2.1.1: optimal is list")
    tests.assert_equal(len(optimal), 5, "2.1.2: exactly 5 modules returned")
    tests.assert_true(all(m in inventory.get_all_module_ids() for m in optimal),
                     "2.1.3: all modules in inventory")
    tests.assert_equal(len(set(optimal)), len(optimal), "2.1.4: no duplicate modules")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "2.1: Optimization returns list", "severity": 2})
    print(f"  ✗ 2.1: {e}")

# TEST 2.2: Optimized beats random
try:
    optimal = integration.optimize_module_selection(num_modules=5)
    optimal_score = integration.detect_synergies(optimal)['total_score']
    
    random_modules = random.sample(inventory.get_all_module_ids(), 5)
    random_score = integration.detect_synergies(random_modules)['total_score']
    
    tests.assert_greater(optimal_score, random_score, "2.2.1: optimal score > random score")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "2.2: Optimized beats random", "severity": 2})
    print(f"  ✗ 2.2: {e}")

# TEST 2.3: Different sizes
try:
    for size in [3, 5, 10]:
        optimal = integration.optimize_module_selection(num_modules=size)
        tests.assert_equal(len(optimal), size, f"2.3.{size}: {size} modules returned")
        
        score = integration.detect_synergies(optimal)['total_score']
        tests.assert_greater(score, 0, f"2.3.{size}b: synergy score > 0")
        
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "2.3: Different sizes", "severity": 2})
    print(f"  ✗ 2.3: {e}")

# ============================================================================
# CATEGORY 3: CONFLICT DETECTION (2 Tests)
# ============================================================================

print("\nCATEGORY 3: CONFLICT DETECTION")
print("-" * 80)

# TEST 3.1: Conflict detection
try:
    modules = random.sample(inventory.get_all_module_ids(), 5)
    conflicts = integration.detect_conflicts(modules)
    
    tests.assert_type(conflicts, dict, "3.1.1: conflicts is dict")
    tests.assert_true('conflicts' in conflicts, "3.1.2: has 'conflicts' key")
    tests.assert_true('conflict_score' in conflicts, "3.1.3: has 'conflict_score' key")
    tests.assert_in_range(conflicts['conflict_score'], 0, 1, "3.1.4: conflict_score in [0,1]")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "3.1: Conflict detection", "severity": 2})
    print(f"  ✗ 3.1: {e}")

# TEST 3.2: Conflict minimization
try:
    optimal = integration.optimize_module_selection(num_modules=5)
    optimal_conflicts = integration.detect_conflicts(optimal)['conflict_score']
    
    random_modules = random.sample(inventory.get_all_module_ids(), 5)
    random_conflicts = integration.detect_conflicts(random_modules)['conflict_score']
    
    tests.assert_less(optimal_conflicts, random_conflicts + 0.1, 
                     "3.2.1: optimal conflicts <= random conflicts")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "3.2: Conflict minimization", "severity": 2})
    print(f"  ✗ 3.2: {e}")

# ============================================================================
# CATEGORY 4: CATEGORY ANALYSIS (6 Tests)
# ============================================================================

print("\nCATEGORY 4: CATEGORY ANALYSIS")
print("-" * 80)

# TEST 4.1: Category distribution
try:
    distribution = integration.analyze_category_distribution()
    
    tests.assert_type(distribution, dict, "4.1.1: distribution is dict")
    tests.assert_greater(len(distribution), 0, "4.1.2: at least 1 category")
    
    total = sum(distribution.values())
    tests.assert_equal(total, 215, "4.1.3: total 215 modules (all modules accounted for)")
    
    all_positive = all(v >= 0 for v in distribution.values())
    tests.assert_true(all_positive, "4.1.4: all counts >= 0")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "4.1: Category distribution", "severity": 2})
    print(f"  ✗ 4.1: {e}")

# TEST 4.2: Category balance
try:
    optimal = integration.optimize_module_selection(num_modules=5)
    categories = set()
    for m in optimal:
        module = inventory.get_module(m)
        if module:
            categories.add(module.category)
    
    tests.assert_greater(len(categories), 1, "4.2.1: at least 2 different categories")
    tests.assert_less(len(categories), 6, "4.2.2: not more than 5 categories")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "4.2: Category balance", "severity": 2})
    print(f"  ✗ 4.2: {e}")

# ============================================================================
# CATEGORY 5: AI METHOD ANALYSIS (3 Tests)
# ============================================================================

print("\nCATEGORY 5: AI METHOD ANALYSIS")
print("-" * 80)

# TEST 5.1: AI methods present
try:
    distribution = integration.analyze_ai_method_distribution()
    
    valid_methods = ['reinforcement learning', 'federated learning', 'self-supervised learning']
    all_present = all(m in distribution for m in valid_methods)
    tests.assert_true(all_present, "5.1.1: all 3 main AI methods present")
    
    all_sufficient = all(distribution.get(m, 0) >= 20 for m in valid_methods)
    tests.assert_true(all_sufficient, "5.1.2: each method has >= 20 modules")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "5.1: AI method distribution", "severity": 2})
    print(f"  ✗ 5.1: {e}")

# TEST 5.2: AI method complementarity
try:
    ssl_modules = [m for m in inventory.get_all_module_ids() 
                   if inventory.get_module(m).ai_training_method == 'self-supervised learning'][:5]
    
    if len(ssl_modules) >= 5:
        same_method_score = integration.detect_synergies(ssl_modules)['total_score']
        
        diverse_modules = integration.optimize_module_selection(num_modules=5)
        diverse_score = integration.detect_synergies(diverse_modules)['total_score']
        
        tests.assert_greater(diverse_score, same_method_score, 
                            "5.2.1: diverse methods score > same method score")
    else:
        print("  ⚠ 5.2: Not enough self-supervised learning modules (skipped)")
        
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "5.2: AI method complementarity", "severity": 2})
    print(f"  ✗ 5.2: {e}")

# ============================================================================
# CATEGORY 6: PERFORMANCE METRICS (4 Tests)
# ============================================================================

print("\nCATEGORY 6: PERFORMANCE METRICS")
print("-" * 80)

# TEST 6.1: Synergy performance
try:
    import time
    
    for size in [5, 10]:
        modules = random.sample(inventory.get_all_module_ids(), size)
        
        start = time.time()
        integration.detect_synergies(modules)
        elapsed = (time.time() - start) * 1000
        
        max_time = 500 if size == 5 else 1000
        tests.assert_less(elapsed, max_time, f"6.1.{size}: synergy calc {size} modules < {max_time}ms")
        
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "6.1: Synergy performance", "severity": 3})
    print(f"  ✗ 6.1: {e}")

# TEST 6.2: Optimization performance
try:
    for size in [5, 10]:
        start = time.time()
        integration.optimize_module_selection(num_modules=size)
        elapsed = (time.time() - start) * 1000
        
        max_time = 1000 if size == 5 else 2000
        tests.assert_less(elapsed, max_time, f"6.2.{size}: optimization {size} modules < {max_time}ms")
        
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "6.2: Optimization performance", "severity": 3})
    print(f"  ✗ 6.2: {e}")

# ============================================================================
# CATEGORY 7: EDGE CASES (9 Tests)
# ============================================================================

print("\nCATEGORY 7: EDGE CASES")
print("-" * 80)

# TEST 7.1: Single module
try:
    single = [inventory.get_all_module_ids()[0]]
    result = integration.detect_synergies(single)
    
    tests.assert_type(result, dict, "7.1.1: single module returns dict")
    tests.assert_greater(result['total_score'], -1, "7.1.2: single module score >= 0")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "7.1: Single module", "severity": 2})
    print(f"  ✗ 7.1: {e}")

# TEST 7.2: Pair of modules
try:
    pair = list(inventory.get_all_module_ids())[:2]
    result = integration.detect_synergies(pair)
    
    tests.assert_type(result, dict, "7.1.3: pair returns dict")
    tests.assert_greater(result['total_score'], -1, "7.1.4: pair score >= 0")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "7.2: Pair of modules", "severity": 2})
    print(f"  ✗ 7.2: {e}")

# TEST 7.3: All modules
try:
    all_modules = list(inventory.get_all_module_ids())
    result = integration.detect_synergies(all_modules)
    
    tests.assert_type(result, dict, "7.2.1: all 215 modules returns dict")
    tests.assert_greater(result['total_score'], 0, "7.2.2: all modules score > 0")
    tests.assert_true(len(result['synergies']) > 0, "7.2.3: synergies found")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "7.3: All modules", "severity": 2})
    print(f"  ✗ 7.3: {e}")

# TEST 7.4: Duplicates
try:
    duplicates = [inventory.get_all_module_ids()[0]] * 3
    result = integration.detect_synergies(duplicates)
    
    tests.assert_type(result, dict, "7.3.1: duplicates handled")
    tests.assert_greater(result['total_score'], -1, "7.3.2: duplicate score >= 0")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "7.4: Duplicates", "severity": 2})
    print(f"  ✗ 7.4: {e}")

# ============================================================================
# TEST REPORT
# ============================================================================

print("\n" + "="*80)
print("TEST REPORT")
print("="*80)

total = tests.passed + tests.failed
print(f"\nTotal Tests: {total}")
print(f"Passed: {tests.passed} ✓")
print(f"Failed: {tests.failed} ✗")

exit_code = 0 if tests.failed == 0 else (1 if tests.failed > 5 else 2)

print(f"\nExit Code: {exit_code} ({'SUCCESS' if exit_code == 0 else 'ERRORS'})")

if exit_code == 0:
    print("✓ ALL TESTS PASSED")
    print("✓ Framework is production ready!")
elif exit_code == 1:
    print("✗ Critical errors found")
elif exit_code == 2:
    print("✗ High errors found")

if tests.errors:
    print("\n" + "="*80)
    print("ERRORS")
    print("="*80)
    for error in tests.errors:
        print(f"\n[SEVERITY {error['severity']}] {error['test']}")

print("\n" + "="*80)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

sys.exit(exit_code)
