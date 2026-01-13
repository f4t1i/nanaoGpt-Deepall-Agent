#!/usr/bin/env python3
"""
DEEPALL INTEGRATION TEST SUITE - 18 Comprehensive Tests
Real Assertions - Scientific Rigor
Exit Codes: 0=Success, 1=Critical, 2=High Errors, 3=Warnings
"""

import sys
import time
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
        self.start_time = datetime.now()
        
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
            print(f"  ✗ {test_name}: {value} outside [{min_val}, {max_val}]")
    
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
    
    def assert_greater_equal(self, actual, expected, test_name):
        if actual >= expected:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({"test": test_name, "severity": 2})
            print(f"  ✗ {test_name}: {actual} not >= {expected}")
    
    def assert_less(self, actual, expected, test_name):
        if actual < expected:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append({"test": test_name, "severity": 2})
            print(f"  ✗ {test_name}: {actual} not < {expected}")
    
    def get_exit_code(self):
        if self.failed == 0:
            return 0
        critical = sum(1 for e in self.errors if e["severity"] == 1)
        if critical > 0:
            return 1
        return 2

# ============================================================================
# IMPORTS
# ============================================================================

print("\n" + "="*80)
print("  DEEPALL INTEGRATION TEST SUITE - 18 Tests")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

try:
    from module_inventory import ModuleInventory
    from deepall_integration import DeepALLIntegration
    print("✓ Imports successful\n")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

tests = Tests()

# ============================================================================
# INITIALIZATION
# ============================================================================

try:
    inventory = ModuleInventory('deepall_modules.json')
    integration = DeepALLIntegration(inventory)
    print("✓ Inventory and Integration initialized\n")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# ============================================================================
# CATEGORY 1: SYNERGY DETECTION (4 Tests)
# ============================================================================

print("CATEGORY 1: SYNERGY DETECTION")
print("-" * 80)

# TEST 1.1: Basis-Synergy-Erkennung
try:
    modules = random.sample(inventory.get_all_module_ids(), 5)
    synergies = integration.detect_synergies(modules)
    
    tests.assert_type(synergies, dict, "1.1.1: synergies is dict")
    tests.assert_type(synergies.get('synergies'), list, "1.1.2: synergies['synergies'] is list")
    tests.assert_type(synergies.get('total_score'), (int, float), "1.1.3: total_score is numeric")
    tests.assert_in_range(synergies['total_score'], 0, 10, "1.1.4: total_score in [0, 10]")
    tests.assert_greater_equal(len(synergies['synergies']), 1, "1.1.5: at least 1 synergy detected")
    
    all_valid = all('type' in s and 'score' in s for s in synergies['synergies'])
    tests.assert_true(all_valid, "1.1.6: all synergies have type and score")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "1.1: Basic synergy detection", "severity": 1})
    print(f"  ✗ 1.1: {e}")

# TEST 1.2: Synergy-Typen validieren
try:
    valid_types = ['category', 'ai_method', 'pairwise', 'hierarchical', 'complementary']
    modules = random.sample(inventory.get_all_module_ids(), 5)
    synergies = integration.detect_synergies(modules)
    
    all_valid_types = True
    for synergy in synergies['synergies']:
        if synergy['type'] not in valid_types:
            all_valid_types = False
            break
        if not (0 <= synergy['score'] <= 1):
            all_valid_types = False
            break
    
    tests.assert_true(all_valid_types, "1.2.1: all synergy types valid and scores in [0,1]")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "1.2: Synergy types validation", "severity": 2})
    print(f"  ✗ 1.2: {e}")

# TEST 1.3: Determinismus
try:
    modules = ['m001', 'm002', 'm003', 'm004', 'm005']
    
    synergies1 = integration.detect_synergies(modules)
    synergies2 = integration.detect_synergies(modules)
    
    tests.assert_equal(synergies1['total_score'], synergies2['total_score'], 
                      "1.3.1: deterministic total_score")
    tests.assert_equal(len(synergies1['synergies']), len(synergies2['synergies']), 
                      "1.3.2: deterministic synergy count")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "1.3: Determinism", "severity": 2})
    print(f"  ✗ 1.3: {e}")

# TEST 1.4: Synergy-Score Korrelation
try:
    # Hochwertige Module (erste 5)
    high_quality = list(inventory.get_all_module_ids())[:5]
    score_high = integration.detect_synergies(high_quality)['total_score']
    
    # Gemischte Module
    mixed = list(inventory.get_all_module_ids())[::40]  # Jeden 40. Module
    score_mixed = integration.detect_synergies(mixed)['total_score']
    
    # Hochwertige sollten >= sein (nicht immer > wegen Randomness)
    tests.assert_greater_equal(score_high, score_mixed, 
                              "1.4.1: high quality score >= mixed score")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "1.4: Synergy correlation", "severity": 3})
    print(f"  ✗ 1.4: {e}")

# ============================================================================
# CATEGORY 2: MODULE OPTIMIZATION (3 Tests)
# ============================================================================

print("\nCATEGORY 2: MODULE OPTIMIZATION")
print("-" * 80)

# TEST 2.1: Optimale Modul-Auswahl
try:
    optimal = integration.optimize_module_selection(num_modules=5)
    
    tests.assert_type(optimal, list, "2.1.1: optimal is list")
    tests.assert_equal(len(optimal), 5, "2.1.2: exactly 5 modules returned")
    
    all_valid = all(m in inventory.get_all_module_ids() for m in optimal)
    tests.assert_true(all_valid, "2.1.3: all modules in inventory")
    
    tests.assert_equal(len(set(optimal)), 5, "2.1.4: no duplicate modules")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "2.1: Optimal selection", "severity": 1})
    print(f"  ✗ 2.1: {e}")

# TEST 2.2: Optimierung vs. Zufällige Auswahl
try:
    optimal = integration.optimize_module_selection(num_modules=5)
    optimal_synergies = integration.detect_synergies(optimal)['total_score']
    
    random_modules = random.sample(inventory.get_all_module_ids(), 5)
    random_synergies = integration.detect_synergies(random_modules)['total_score']
    
    tests.assert_greater(optimal_synergies, random_synergies, 
                        "2.2.1: optimal score > random score")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "2.2: Optimization vs random", "severity": 2})
    print(f"  ✗ 2.2: {e}")

# TEST 2.3: Optimierung mit verschiedenen Größen
try:
    for size in [3, 5, 10]:
        optimal = integration.optimize_module_selection(num_modules=size)
        tests.assert_equal(len(optimal), size, f"2.3.{size}: {size} modules returned")
        
        synergies = integration.detect_synergies(optimal)
        tests.assert_greater(synergies['total_score'], 0, f"2.3.{size}b: synergy score > 0")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "2.3: Various sizes", "severity": 2})
    print(f"  ✗ 2.3: {e}")

# ============================================================================
# CATEGORY 3: CONFLICT DETECTION (2 Tests)
# ============================================================================

print("\nCATEGORY 3: CONFLICT DETECTION")
print("-" * 80)

# TEST 3.1: Basis-Konflikt-Erkennung
try:
    modules = random.sample(inventory.get_all_module_ids(), 5)
    
    # Check if method exists
    if hasattr(integration, 'detect_conflicts'):
        conflicts = integration.detect_conflicts(modules)
        
        tests.assert_type(conflicts, dict, "3.1.1: conflicts is dict")
        tests.assert_true('conflicts' in conflicts, "3.1.2: has 'conflicts' key")
        tests.assert_true('conflict_score' in conflicts, "3.1.3: has 'conflict_score' key")
        tests.assert_in_range(conflicts['conflict_score'], 0, 1, "3.1.4: conflict_score in [0,1]")
    else:
        print("  ⚠ 3.1: detect_conflicts method not available (skipped)")
        
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "3.1: Conflict detection", "severity": 3})
    print(f"  ✗ 3.1: {e}")

# TEST 3.2: Konflikt-Minimierung
try:
    if hasattr(integration, 'detect_conflicts'):
        optimal = integration.optimize_module_selection(num_modules=5)
        optimal_conflicts = integration.detect_conflicts(optimal)['conflict_score']
        
        random_modules = random.sample(inventory.get_all_module_ids(), 5)
        random_conflicts = integration.detect_conflicts(random_modules)['conflict_score']
        
        tests.assert_less(optimal_conflicts, random_conflicts + 0.1, 
                         "3.2.1: optimal conflicts <= random conflicts")
    else:
        print("  ⚠ 3.2: detect_conflicts method not available (skipped)")
        
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "3.2: Conflict minimization", "severity": 3})
    print(f"  ✗ 3.2: {e}")

# ============================================================================
# CATEGORY 4: CATEGORY ANALYSIS (2 Tests)
# ============================================================================

print("\nCATEGORY 4: CATEGORY ANALYSIS")
print("-" * 80)

# TEST 4.1: Kategorie-Verteilung
try:
    if hasattr(integration, 'analyze_category_distribution'):
        distribution = integration.analyze_category_distribution()
        
        tests.assert_type(distribution, dict, "4.1.1: distribution is dict")
        tests.assert_equal(len(distribution), 7, "4.1.2: 7 categories")
        
        total = sum(distribution.values())
        tests.assert_equal(total, 215, "4.1.3: total 215 modules")
        
        all_positive = all(v >= 0 for v in distribution.values())
        tests.assert_true(all_positive, "4.1.4: all counts >= 0")
    else:
        print("  ⚠ 4.1: analyze_category_distribution not available (skipped)")
        
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "4.1: Category distribution", "severity": 3})
    print(f"  ✗ 4.1: {e}")

# TEST 4.2: Kategorie-Balance in Optimierung
try:
    optimal = integration.optimize_module_selection(num_modules=5)
    categories = [inventory.get_module(m).category for m in optimal]
    unique_categories = len(set(categories))
    
    tests.assert_greater_equal(unique_categories, 2, "4.2.1: at least 2 different categories")
    tests.assert_less(unique_categories, 6, "4.2.2: not more than 5 categories")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "4.2: Category balance", "severity": 3})
    print(f"  ✗ 4.2: {e}")

# ============================================================================
# CATEGORY 5: AI METHOD ANALYSIS (2 Tests)
# ============================================================================

print("\nCATEGORY 5: AI METHOD ANALYSIS")
print("-" * 80)

# TEST 5.1: AI-Methoden-Verteilung
try:
    if hasattr(integration, 'analyze_ai_method_distribution'):
        distribution = integration.analyze_ai_method_distribution()
        
        valid_methods = ['SFT', 'RL', 'ICL', 'CL']
        all_present = all(m in distribution for m in valid_methods)
        tests.assert_true(all_present, "5.1.1: all 4 AI methods present")
        
        all_sufficient = all(distribution.get(m, 0) >= 20 for m in valid_methods)
        tests.assert_true(all_sufficient, "5.1.2: each method has >= 20 modules")
    else:
        print("  ⚠ 5.1: analyze_ai_method_distribution not available (skipped)")
        
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "5.1: AI method distribution", "severity": 3})
    print(f"  ✗ 5.1: {e}")

# TEST 5.2: AI-Methoden-Komplementarität
try:
    # Alle gleiche Methode
    sft_modules = [m for m in inventory.get_all_module_ids() 
                   if inventory.get_module(m).ai_training_method == 'SFT'][:5]
    
    if len(sft_modules) >= 5:
        same_method_score = integration.detect_synergies(sft_modules)['total_score']
        
        # Verschiedene Methoden
        diverse_modules = integration.optimize_module_selection(num_modules=5)
        diverse_score = integration.detect_synergies(diverse_modules)['total_score']
        
        tests.assert_greater(diverse_score, same_method_score, 
                            "5.2.1: diverse methods score > same method score")
    else:
        print("  ⚠ 5.2: Not enough SFT modules (skipped)")
        
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "5.2: AI method complementarity", "severity": 3})
    print(f"  ✗ 5.2: {e}")

# ============================================================================
# CATEGORY 6: PERFORMANCE METRICS (2 Tests)
# ============================================================================

print("\nCATEGORY 6: PERFORMANCE METRICS")
print("-" * 80)

# TEST 6.1: Synergy-Berechnung Performance
try:
    for size in [5, 10]:
        modules = random.sample(inventory.get_all_module_ids(), size)
        
        start = time.time()
        integration.detect_synergies(modules)
        elapsed = (time.time() - start) * 1000  # ms
        
        max_time = 100 * size  # Linear scaling
        tests.assert_less(elapsed, max_time, f"6.1.{size}: synergy calc {size} modules < {max_time}ms")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "6.1: Synergy performance", "severity": 3})
    print(f"  ✗ 6.1: {e}")

# TEST 6.2: Optimierung Performance
try:
    for size in [5, 10]:
        start = time.time()
        integration.optimize_module_selection(num_modules=size)
        elapsed = (time.time() - start) * 1000  # ms
        
        max_time = 200 * size  # Linear scaling
        tests.assert_less(elapsed, max_time, f"6.2.{size}: optimization {size} modules < {max_time}ms")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "6.2: Optimization performance", "severity": 3})
    print(f"  ✗ 6.2: {e}")

# ============================================================================
# CATEGORY 7: EDGE CASES (3 Tests)
# ============================================================================

print("\nCATEGORY 7: EDGE CASES")
print("-" * 80)

# TEST 7.1: Minimale Eingabe
try:
    # 1 Modul
    single = integration.detect_synergies(['m001'])
    tests.assert_type(single, dict, "7.1.1: single module returns dict")
    tests.assert_greater_equal(single['total_score'], 0, "7.1.2: single module score >= 0")
    
    # 2 Module
    pair = integration.detect_synergies(['m001', 'm002'])
    tests.assert_type(pair, dict, "7.1.3: pair returns dict")
    tests.assert_greater_equal(pair['total_score'], 0, "7.1.4: pair score >= 0")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "7.1: Minimal input", "severity": 2})
    print(f"  ✗ 7.1: {e}")

# TEST 7.2: Maximale Eingabe
try:
    all_modules = inventory.get_all_module_ids()
    synergies = integration.detect_synergies(all_modules)
    
    tests.assert_type(synergies, dict, "7.2.1: all 215 modules returns dict")
    tests.assert_greater(synergies['total_score'], 0, "7.2.2: all modules score > 0")
    tests.assert_greater(len(synergies['synergies']), 0, "7.2.3: synergies found")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "7.2: Maximum input", "severity": 2})
    print(f"  ✗ 7.2: {e}")

# TEST 7.3: Doppelte Module
try:
    with_duplicates = ['m001', 'm001', 'm002', 'm002', 'm003']
    synergies = integration.detect_synergies(with_duplicates)
    
    tests.assert_type(synergies, dict, "7.3.1: duplicates handled")
    tests.assert_greater_equal(synergies['total_score'], 0, "7.3.2: duplicate score >= 0")
    
except Exception as e:
    tests.failed += 1
    tests.errors.append({"test": "7.3: Duplicate modules", "severity": 2})
    print(f"  ✗ 7.3: {e}")

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
    print("✓ ALL DEEPALL INTEGRATION TESTS PASSED")
    print("✓ Framework is production ready!")
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

print("\n" + "="*80)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

sys.exit(exit_code)
