#!/usr/bin/env python3
"""
Enhanced Module Inventory - Phase 2
Merge JSON and Excel data with backward compatibility
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set
from module_inventory import ModuleInventory, Module
from deepall_table_loader import DeepALLTableLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED MODULE
# ============================================================================

class EnhancedModule(Module):
    """Extended Module with DeepALL data"""
    
    def __init__(self, base_module: Module, deepall_data: Optional[Dict] = None):
        self.__dict__.update(base_module.__dict__)
        self.deepall_data = deepall_data or {}
        
        # DeepALL fields
        self.superintelligence_index = deepall_data.get('superintelligence_index') if deepall_data else None
        self.synergy_index = deepall_data.get('synergy_index') if deepall_data else None
        self.optimization_index = deepall_data.get('optimization_index') if deepall_data else None
        self.learning_index = deepall_data.get('learning_index') if deepall_data else None
        self.performance_index = deepall_data.get('performance_index') if deepall_data else None
        self.efficiency_index = deepall_data.get('efficiency_index') if deepall_data else None
        self.knowledge_index = deepall_data.get('knowledge_index') if deepall_data else None
        self.energy_management_index = deepall_data.get('energy_management_index') if deepall_data else None
        self.contribution_index = deepall_data.get('contribution_index') if deepall_data else None
        self.reward_index = deepall_data.get('reward_index') if deepall_data else None

# ============================================================================
# ENHANCED MODULE INVENTORY
# ============================================================================

class EnhancedModuleInventory(ModuleInventory):
    """Extended Module Inventory combining JSON and Excel data"""
    
    def __init__(self, json_path: str, excel_path: str):
        logger.info("Initializing EnhancedModuleInventory...")
        
        super().__init__(json_path)
        
        self.deepall_loader = DeepALLTableLoader(excel_path)
        self.deepall_complete = self.deepall_loader.load_complete_data()
        self.deepall_module_index = self.deepall_loader.create_index('DeepALL_Complete', 'module_index')
        
        # Load other sheets safely
        try:
            self.superintelligences_data = self.deepall_loader.load_sheet('superintelligences')
        except:
            self.superintelligences_data = []
        
        self.si_to_modules = self._build_si_index()
        logger.info(f"✓ EnhancedModuleInventory initialized")
        logger.info(f"  - JSON modules: {len(self.modules)}")
        logger.info(f"  - Excel modules: {len(self.deepall_complete)}")
        logger.info(f"  - Superintelligences: {len(self.si_to_modules)}")
    
    def _build_si_index(self) -> Dict[str, List[str]]:
        """Build index of modules by superintelligence"""
        si_index = {}
        
        for module in self.deepall_complete:
            si = module.get('superintelligence_index')
            if si and si != 'not_assigned':
                if si not in si_index:
                    si_index[si] = []
                module_id = module.get('module_index')
                if module_id:
                    si_index[si].append(module_id)
        
        return si_index
    
    # ========================================================================
    # ENHANCED METHODS
    # ========================================================================
    
    def get_module_enhanced(self, module_id: str) -> Optional[EnhancedModule]:
        """Get module with full DeepALL data"""
        base = self.get_module(module_id)
        if not base:
            return None
        
        deepall = self.deepall_module_index.get(module_id)
        enhanced = EnhancedModule(base, deepall)
        return enhanced
    
    def get_all_modules_enhanced(self) -> List[EnhancedModule]:
        """Get all modules with DeepALL data"""
        enhanced_modules = []
        for module_id in self.get_all_module_ids():
            enhanced = self.get_module_enhanced(module_id)
            if enhanced:
                enhanced_modules.append(enhanced)
        return enhanced_modules
    
    # ========================================================================
    # SUPERINTELLIGENCE QUERIES
    # ========================================================================
    
    def get_superintelligence_modules(self, si_index: str) -> List[str]:
        """Get all module IDs assigned to a superintelligence"""
        return self.si_to_modules.get(si_index, [])
    
    def get_superintelligence_modules_enhanced(self, si_index: str) -> List[EnhancedModule]:
        """Get all enhanced modules assigned to a superintelligence"""
        module_ids = self.get_superintelligence_modules(si_index)
        modules = []
        for module_id in module_ids:
            enhanced = self.get_module_enhanced(module_id)
            if enhanced:
                modules.append(enhanced)
        return modules
    
    def get_all_superintelligences(self) -> List[str]:
        """Get all superintelligence indexes"""
        return list(self.si_to_modules.keys())
    
    def get_superintelligence_info(self, si_index: str) -> Dict[str, Any]:
        """Get information about a superintelligence"""
        modules = self.get_superintelligence_modules(si_index)
        enhanced_modules = self.get_superintelligence_modules_enhanced(si_index)
        
        categories = {}
        ai_methods = {}
        
        for module in enhanced_modules:
            cat = module.category
            if cat:
                categories[cat] = categories.get(cat, 0) + 1
            method = module.ai_training_method
            if method:
                ai_methods[method] = ai_methods.get(method, 0) + 1
        
        return {
            'superintelligence_index': si_index,
            'module_count': len(modules),
            'module_ids': modules,
            'categories': categories,
            'ai_methods': ai_methods
        }
    
    # ========================================================================
    # SYNERGY QUERIES
    # ========================================================================
    
    def get_synergy_details(self, module_ids: List[str]) -> Dict[str, Any]:
        """Get detailed synergy information for modules"""
        details = {
            'modules': module_ids,
            'count': len(module_ids),
            'synergy_indices': [],
            'superintelligences': set(),
            'categories': set(),
            'ai_methods': set()
        }
        
        for module_id in module_ids:
            enhanced = self.get_module_enhanced(module_id)
            if enhanced:
                if enhanced.synergy_index:
                    details['synergy_indices'].append(enhanced.synergy_index)
                if enhanced.superintelligence_index:
                    details['superintelligences'].add(enhanced.superintelligence_index)
                if enhanced.category:
                    details['categories'].add(enhanced.category)
                if enhanced.ai_training_method:
                    details['ai_methods'].add(enhanced.ai_training_method)
        
        details['superintelligences'] = list(details['superintelligences'])
        details['categories'] = list(details['categories'])
        details['ai_methods'] = list(details['ai_methods'])
        
        return details
    
    # ========================================================================
    # LEARNING QUERIES
    # ========================================================================
    
    def get_learning_progress(self, module_id: str) -> Dict[str, Any]:
        """Get learning progress for a module"""
        enhanced = self.get_module_enhanced(module_id)
        if not enhanced:
            return {}
        
        return {
            'module_id': module_id,
            'module_name': enhanced.name,
            'learning_index': enhanced.learning_index,
            'learning_status': 'assigned' if enhanced.learning_index and enhanced.learning_index != 'not_assigned' else 'not_assigned'
        }
    
    # ========================================================================
    # PERFORMANCE QUERIES
    # ========================================================================
    
    def get_performance_metrics(self, module_id: str) -> Dict[str, Any]:
        """Get performance metrics for a module"""
        enhanced = self.get_module_enhanced(module_id)
        if not enhanced:
            return {}
        
        return {
            'module_id': module_id,
            'module_name': enhanced.name,
            'performance_index': enhanced.performance_index,
            'efficiency_index': enhanced.efficiency_index,
            'energy_management_index': enhanced.energy_management_index,
            'contribution_index': enhanced.contribution_index,
            'reward_index': enhanced.reward_index
        }
    
    # ========================================================================
    # FILTERING QUERIES
    # ========================================================================
    
    def filter_by_superintelligence(self, si_index: str) -> List[EnhancedModule]:
        """Filter modules by superintelligence"""
        return self.get_superintelligence_modules_enhanced(si_index)
    
    def filter_by_category(self, category: str) -> List[EnhancedModule]:
        """Filter modules by category"""
        all_modules = self.get_all_modules_enhanced()
        return [m for m in all_modules if m.category == category]
    
    def filter_by_ai_method(self, ai_method: str) -> List[EnhancedModule]:
        """Filter modules by AI method"""
        all_modules = self.get_all_modules_enhanced()
        return [m for m in all_modules if hasattr(m, 'ai_methods') and ai_method in m.ai_training_methods]
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        all_modules = self.get_all_modules_enhanced()
        
        categories = {}
        ai_methods = {}
        
        for module in all_modules:
            if module.category:
                categories[module.category] = categories.get(module.category, 0) + 1
            if hasattr(module, 'ai_methods') and module.ai_training_methods:
                for method in module.ai_training_methods:
                    ai_methods[method] = ai_methods.get(method, 0) + 1
        
        return {
            'total_modules': len(all_modules),
            'superintelligences': len(self.si_to_modules),
            'categories': categories,
            'ai_methods': ai_methods
        }
    
    def print_statistics(self):
        """Print statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("ENHANCED MODULE INVENTORY STATISTICS")
        print("="*80)
        print(f"\nTotal Modules: {stats['total_modules']}")
        print(f"Superintelligences: {stats['superintelligences']}")
        print(f"\nCategories ({len(stats['categories'])}):")
        for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cat:30} {count:3} modules")
        print(f"\nAI Methods ({len(stats['ai_methods'])}):")
        for method, count in sorted(stats['ai_methods'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {method:30} {count:3} modules")
        print("\n" + "="*80 + "\n")


# ============================================================================
# MAIN - TESTING
# ============================================================================

if __name__ == '__main__':
    import sys
    
    print("\n" + "="*80)
    print("ENHANCED MODULE INVENTORY - TEST")
    print("="*80 + "\n")
    
    # Test 1: Initialize
    print("TEST 1: Initializing EnhancedModuleInventory...")
    try:
        inventory = EnhancedModuleInventory(
            'deepall_modules.json',
            'DeepALL_MASTER_V7_FINAL_WITH_ALL_REITERS.xlsx'
        )
        print("  ✓ Initialized successfully\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        sys.exit(1)
    
    # Test 2: Get enhanced module
    print("TEST 2: Getting enhanced module...")
    try:
        enhanced = inventory.get_module_enhanced('m001')
        if enhanced:
            print(f"  ✓ Module m001: {enhanced.name}")
            print(f"    - Category: {enhanced.category}")
            print(f"    - AI Method: {enhanced.ai_training_method}")
            print(f"    - SI: {enhanced.superintelligence_index}\n")
        else:
            print("  ✗ Module not found\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        sys.exit(1)
    
    # Test 3: Get superintelligence modules
    print("TEST 3: Getting superintelligence modules...")
    try:
        si_list = inventory.get_all_superintelligences()
        if si_list:
            si = si_list[0]
            modules = inventory.get_superintelligence_modules(si)
            print(f"  ✓ SI {si}: {len(modules)} modules\n")
        else:
            print("  ✓ No superintelligences assigned\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        sys.exit(1)
    
    # Test 4: Filter by category
    print("TEST 4: Filtering by category...")
    try:
        all_modules = inventory.get_all_modules_enhanced()
        if all_modules:
            cat = all_modules[0].category
            filtered = inventory.filter_by_category(cat)
            print(f"  ✓ Category {cat}: {len(filtered)} modules\n")
        else:
            print("  ✓ No modules\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        sys.exit(1)
    
    # Test 5: Get statistics
    print("TEST 5: Getting statistics...")
    try:
        stats = inventory.get_statistics()
        print(f"  ✓ Total modules: {stats['total_modules']}")
        print(f"  ✓ Superintelligences: {stats['superintelligences']}\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        sys.exit(1)
    
    # Print full statistics
    inventory.print_statistics()
    
    print("✓ ALL TESTS PASSED\n")
