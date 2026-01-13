#!/usr/bin/env python3
"""
Kaggle Test Script - Import and Test All Modules
Run this script in Kaggle to verify all modules work
"""

import sys
import os
from datetime import datetime

print("=" * 80)
print("  nanoGPT-DeepALL-Agent: Kaggle Test Script")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python Version: {sys.version}")
print(f"Working Directory: {os.getcwd()}")
print()

# ============================================================================
# STEP 1: Import All Modules
# ============================================================================
print("STEP 1: Importing All Modules")
print("-" * 80)

modules_imported = []
modules_failed = []

modules_to_import = [
    ('module_inventory', 'ModuleInventory'),
    ('reward_system', 'RewardSystem'),
    ('reward_system', 'ExecutionResult'),
    ('data_generator', 'TrainingDataGenerator'),
    ('sft_trainer', 'SFTTrainer'),
    ('rl_trainer', 'RLTrainer'),
    ('icl_trainer', 'ICLTrainer'),
    ('continuous_learning_trainer', 'ContinuousLearningTrainer'),
    ('deepall_integration', 'DeepALLIntegration'),
    ('training_orchestrator', 'TrainingOrchestrator'),
]

for module_name, class_name in modules_to_import:
    try:
        exec(f"from {module_name} import {class_name}")
        modules_imported.append((module_name, class_name))
        print(f"✓ {module_name:40} → {class_name}")
    except Exception as e:
        modules_failed.append((module_name, class_name, str(e)))
        print(f"✗ {module_name:40} → {class_name}: {str(e)[:50]}")

print()
print(f"Imported: {len(modules_imported)}/{len(modules_to_import)}")

if modules_failed:
    print(f"Failed: {len(modules_failed)}")
    for module_name, class_name, error in modules_failed:
        print(f"  - {module_name}.{class_name}: {error}")
    sys.exit(1)

print()

# ============================================================================
# STEP 2: Quick Functionality Tests
# ============================================================================
print("STEP 2: Quick Functionality Tests")
print("-" * 80)

try:
    # Test 1: Load Module Inventory
    print("\n[1/5] Testing Module Inventory...")
    from module_inventory import ModuleInventory
    inventory = ModuleInventory('deepall_modules.json')
    num_modules = len(inventory.modules)
    print(f"✓ Loaded {num_modules} modules")
    
    # Test 2: Reward System
    print("\n[2/5] Testing Reward System...")
    from reward_system import RewardSystem, ExecutionResult
    reward_system = RewardSystem()
    r = ExecutionResult("test_task", ["m001"], ["m001"], 5.0, 0.3, True)
    reward = reward_system.calculate_reward(r)
    print(f"✓ Reward calculated: {reward:.4f}")
    
    # Test 3: Data Generator
    print("\n[3/5] Testing Data Generator...")
    from data_generator import TrainingDataGenerator
    generator = TrainingDataGenerator(inventory)
    sft_data = generator.generate_dataset(num_samples=10)
    print(f"✓ Generated {len(sft_data)} SFT examples")
    
    # Test 4: SFT Trainer
    print("\n[4/5] Testing SFT Trainer...")
    from sft_trainer import SFTTrainer
    sft_trainer = SFTTrainer(inventory)
    sft_trainer.generate_training_data(num_samples=10)
    print(f"✓ SFT Trainer initialized with {len(sft_trainer.training_data)} samples")
    
    # Test 5: DeepALL Integration
    print("\n[5/5] Testing DeepALL Integration...")
    from deepall_integration import DeepALLIntegration
    integration = DeepALLIntegration(inventory)
    import random
    sample = random.sample(inventory.get_all_module_ids(), min(3, len(inventory.get_all_module_ids())))
    synergies = integration.detect_synergies(sample)
    print(f"✓ Detected synergies: {len(synergies['synergies'])}")
    
    print("\n✓ All functionality tests passed!")
    
except Exception as e:
    print(f"\n✗ Functionality test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================================
# STEP 3: Summary
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ All {len(modules_imported)} modules imported successfully")
print(f"✓ All functionality tests passed")
print(f"✓ Module Inventory: {num_modules} modules loaded")
print(f"✓ Reward System: Working")
print(f"✓ Data Generator: Working")
print(f"✓ SFT Trainer: Working")
print(f"✓ DeepALL Integration: Working")
print()
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("✓ KAGGLE TEST SUCCESSFUL - Framework is ready to use!")
print("=" * 80)
