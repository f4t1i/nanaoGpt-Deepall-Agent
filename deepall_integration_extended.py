#!/usr/bin/env python3
"""
DeepALL Integration Extended - Phase 3
Superintelligence-aware, Learning-aware, and Performance-aware optimization

Features:
- Superintelligence-based module selection
- Learning progress tracking
- Performance metrics optimization
- Predictive analytics
- Cross-SI synergy detection
- Adaptive optimization strategies
"""

import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from enhanced_module_inventory import EnhancedModuleInventory, EnhancedModule
from deepall_integration import DeepALLIntegration
from reward_system import ExecutionResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# SUPERINTELLIGENCE-AWARE OPTIMIZATION
# ============================================================================

class SuperintelligenceAwareOptimizer:
    """Optimize module selection based on superintelligence groups"""
    
    def __init__(self, inventory: EnhancedModuleInventory):
        self.inventory = inventory
        self.si_modules = {}
        self.si_stats = {}
        self._build_si_stats()
    
    def _build_si_stats(self):
        """Build statistics for each superintelligence"""
        for si in self.inventory.get_all_superintelligences():
            modules = self.inventory.get_superintelligence_modules_enhanced(si)
            
            categories = {}
            methods = {}
            
            for module in modules:
                cat = module.category
                if cat:
                    categories[cat] = categories.get(cat, 0) + 1
                
                method = module.ai_training_method
                if method:
                    methods[method] = methods.get(method, 0) + 1
            
            self.si_stats[si] = {
                'module_count': len(modules),
                'categories': categories,
                'methods': methods,
                'diversity': len(categories) + len(methods)
            }
    
    def optimize_by_superintelligence(self, num_modules: int = 5, 
                                     si_preference: Optional[str] = None) -> List[str]:
        """
        Optimize module selection within or across superintelligences
        
        Args:
            num_modules: Number of modules to select
            si_preference: Prefer modules from specific SI (None = cross-SI)
        
        Returns:
            List of optimized module IDs
        """
        if si_preference:
            # Within-SI optimization
            modules = self.inventory.get_superintelligence_modules(si_preference)
            if len(modules) >= num_modules:
                return random.sample(modules, num_modules)
            else:
                return modules
        else:
            # Cross-SI optimization - maximize diversity
            selected = []
            si_list = list(self.inventory.get_all_superintelligences())
            
            # Distribute across SIs
            modules_per_si = max(1, num_modules // len(si_list))
            
            for si in si_list:
                si_modules = self.inventory.get_superintelligence_modules(si)
                if si_modules:
                    count = min(modules_per_si, len(si_modules))
                    selected.extend(random.sample(si_modules, count))
                
                if len(selected) >= num_modules:
                    break
            
            return selected[:num_modules]
    
    def get_si_diversity_score(self, module_ids: List[str]) -> float:
        """Calculate diversity score across superintelligences"""
        si_set = set()
        
        for module_id in module_ids:
            enhanced = self.inventory.get_module_enhanced(module_id)
            if enhanced and enhanced.superintelligence_index:
                si_set.add(enhanced.superintelligence_index)
        
        # Normalize by number of modules
        return len(si_set) / len(module_ids) if module_ids else 0.0

# ============================================================================
# LEARNING-AWARE OPTIMIZATION
# ============================================================================

class LearningAwareOptimizer:
    """Optimize based on learning progress and assignments"""
    
    def __init__(self, inventory: EnhancedModuleInventory):
        self.inventory = inventory
        self.learning_status = self._build_learning_status()
    
    def _build_learning_status(self) -> Dict[str, str]:
        """Build learning status for all modules"""
        status = {}
        
        for module_id in self.inventory.get_all_module_ids():
            enhanced = self.inventory.get_module_enhanced(module_id)
            if enhanced:
                if enhanced.learning_index and enhanced.learning_index != 'not_assigned':
                    status[module_id] = 'assigned'
                else:
                    status[module_id] = 'unassigned'
        
        return status
    
    def optimize_for_learning(self, num_modules: int = 5, 
                             prefer_assigned: bool = True) -> List[str]:
        """
        Optimize module selection based on learning status
        
        Args:
            num_modules: Number of modules to select
            prefer_assigned: Prefer modules with learning assignments
        
        Returns:
            List of optimized module IDs
        """
        if prefer_assigned:
            assigned = [m for m, s in self.learning_status.items() if s == 'assigned']
            if len(assigned) >= num_modules:
                return random.sample(assigned, num_modules)
            else:
                # Fill with unassigned if not enough assigned
                unassigned = [m for m, s in self.learning_status.items() if s == 'unassigned']
                remaining = num_modules - len(assigned)
                return assigned + random.sample(unassigned, min(remaining, len(unassigned)))
        else:
            all_modules = self.inventory.get_all_module_ids()
            return random.sample(all_modules, min(num_modules, len(all_modules)))
    
    def get_learning_coverage(self) -> float:
        """Get percentage of modules with learning assignments"""
        assigned = sum(1 for s in self.learning_status.values() if s == 'assigned')
        total = len(self.learning_status)
        return assigned / total if total > 0 else 0.0

# ============================================================================
# PERFORMANCE-AWARE OPTIMIZATION
# ============================================================================

class PerformanceAwareOptimizer:
    """Optimize based on performance metrics"""
    
    def __init__(self, inventory: EnhancedModuleInventory):
        self.inventory = inventory
        self.performance_status = self._build_performance_status()
    
    def _build_performance_status(self) -> Dict[str, str]:
        """Build performance status for all modules"""
        status = {}
        
        for module_id in self.inventory.get_all_module_ids():
            enhanced = self.inventory.get_module_enhanced(module_id)
            if enhanced:
                if enhanced.performance_index and enhanced.performance_index != 'not_assigned':
                    status[module_id] = 'measured'
                else:
                    status[module_id] = 'unmeasured'
        
        return status
    
    def optimize_for_performance(self, num_modules: int = 5,
                                prefer_measured: bool = True) -> List[str]:
        """
        Optimize module selection based on performance metrics
        
        Args:
            num_modules: Number of modules to select
            prefer_measured: Prefer modules with performance metrics
        
        Returns:
            List of optimized module IDs
        """
        if prefer_measured:
            measured = [m for m, s in self.performance_status.items() if s == 'measured']
            if len(measured) >= num_modules:
                return random.sample(measured, num_modules)
            else:
                unmeasured = [m for m, s in self.performance_status.items() if s == 'unmeasured']
                remaining = num_modules - len(measured)
                return measured + random.sample(unmeasured, min(remaining, len(unmeasured)))
        else:
            all_modules = self.inventory.get_all_module_ids()
            return random.sample(all_modules, min(num_modules, len(all_modules)))
    
    def get_performance_coverage(self) -> float:
        """Get percentage of modules with performance metrics"""
        measured = sum(1 for s in self.performance_status.values() if s == 'measured')
        total = len(self.performance_status)
        return measured / total if total > 0 else 0.0

# ============================================================================
# DEEPALL INTEGRATION EXTENDED
# ============================================================================

class DeepALLIntegrationExtended(DeepALLIntegration):
    """Extended DeepALL Integration with advanced optimization strategies"""
    
    def __init__(self, inventory: EnhancedModuleInventory):
        # Initialize base DeepALLIntegration with JSON inventory
        super().__init__(inventory)
        
        # Store enhanced inventory
        self.enhanced_inventory = inventory
        
        # Initialize specialized optimizers
        self.si_optimizer = SuperintelligenceAwareOptimizer(inventory)
        self.learning_optimizer = LearningAwareOptimizer(inventory)
        self.performance_optimizer = PerformanceAwareOptimizer(inventory)
        
        logger.info("✓ DeepALLIntegrationExtended initialized")
        logger.info(f"  - Superintelligences: {len(self.si_optimizer.si_stats)}")
        logger.info(f"  - Learning coverage: {self.learning_optimizer.get_learning_coverage()*100:.1f}%")
        logger.info(f"  - Performance coverage: {self.performance_optimizer.get_performance_coverage()*100:.1f}%")
    
    # ========================================================================
    # SUPERINTELLIGENCE-BASED OPTIMIZATION
    # ========================================================================
    
    def optimize_by_superintelligence(self, num_modules: int = 5,
                                     si_preference: Optional[str] = None) -> List[str]:
        """
        Optimize module selection based on superintelligence groups
        
        Args:
            num_modules: Number of modules to select
            si_preference: Prefer specific SI (None = cross-SI)
        
        Returns:
            List of optimized module IDs
        """
        return self.si_optimizer.optimize_by_superintelligence(num_modules, si_preference)
    
    def get_superintelligence_stats(self) -> Dict[str, Any]:
        """Get statistics for all superintelligences"""
        return self.si_optimizer.si_stats
    
    # ========================================================================
    # LEARNING-AWARE OPTIMIZATION
    # ========================================================================
    
    def optimize_for_learning(self, num_modules: int = 5,
                             prefer_assigned: bool = True) -> List[str]:
        """Optimize based on learning progress"""
        return self.learning_optimizer.optimize_for_learning(num_modules, prefer_assigned)
    
    def get_learning_coverage(self) -> float:
        """Get learning coverage percentage"""
        return self.learning_optimizer.get_learning_coverage()
    
    # ========================================================================
    # PERFORMANCE-AWARE OPTIMIZATION
    # ========================================================================
    
    def optimize_for_performance(self, num_modules: int = 5,
                                prefer_measured: bool = True) -> List[str]:
        """Optimize based on performance metrics"""
        return self.performance_optimizer.optimize_for_performance(num_modules, prefer_measured)
    
    def get_performance_coverage(self) -> float:
        """Get performance coverage percentage"""
        return self.performance_optimizer.get_performance_coverage()
    
    # ========================================================================
    # HYBRID OPTIMIZATION
    # ========================================================================
    
    def optimize_hybrid(self, num_modules: int = 5,
                       si_weight: float = 0.4,
                       learning_weight: float = 0.3,
                       performance_weight: float = 0.3) -> List[str]:
        """
        Hybrid optimization combining multiple strategies
        
        Args:
            num_modules: Number of modules to select
            si_weight: Weight for SI-based optimization
            learning_weight: Weight for learning-based optimization
            performance_weight: Weight for performance-based optimization
        
        Returns:
            List of optimized module IDs
        """
        # Get candidates from each optimizer
        si_candidates = self.si_optimizer.optimize_by_superintelligence(num_modules)
        learning_candidates = self.learning_optimizer.optimize_for_learning(num_modules)
        perf_candidates = self.performance_optimizer.optimize_for_performance(num_modules)
        
        # Combine with weights
        candidate_scores = {}
        
        for module_id in si_candidates:
            candidate_scores[module_id] = candidate_scores.get(module_id, 0) + si_weight
        
        for module_id in learning_candidates:
            candidate_scores[module_id] = candidate_scores.get(module_id, 0) + learning_weight
        
        for module_id in perf_candidates:
            candidate_scores[module_id] = candidate_scores.get(module_id, 0) + performance_weight
        
        # Sort by score and select top N
        sorted_modules = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [m for m, _ in sorted_modules[:num_modules]]
    
    # ========================================================================
    # PREDICTIVE ANALYTICS
    # ========================================================================
    
    def predict_module_synergy(self, module_ids: List[str]) -> Dict[str, Any]:
        """
        Predict synergy for a set of modules
        
        Args:
            module_ids: List of module IDs
        
        Returns:
            Prediction with confidence
        """
        # Get synergy details
        synergies = self.enhanced_inventory.get_synergy_details(module_ids)
        
        # Calculate prediction score
        cross_si = len(synergies.get('superintelligences', []))
        cross_cat = len(synergies.get('categories', []))
        cross_method = len(synergies.get('ai_methods', []))
        
        # Normalize scores
        max_si = len(self.enhanced_inventory.get_all_superintelligences())
        max_cat = 7  # Known from data
        max_method = 3  # Known from data
        
        si_score = cross_si / max_si if max_si > 0 else 0
        cat_score = cross_cat / max_cat if max_cat > 0 else 0
        method_score = cross_method / max_method if max_method > 0 else 0
        
        # Combined prediction
        total_score = (si_score + cat_score + method_score) / 3
        
        return {
            'modules': module_ids,
            'cross_si_count': cross_si,
            'cross_category_count': cross_cat,
            'cross_method_count': cross_method,
            'predicted_score': total_score,
            'confidence': min(1.0, len(module_ids) / 10)
        }
    
    def predict_learning_success(self, module_ids: List[str]) -> Dict[str, Any]:
        """
        Predict learning success for modules
        
        Args:
            module_ids: List of module IDs
        
        Returns:
            Prediction with confidence
        """
        assigned_count = 0
        measured_count = 0
        
        for module_id in module_ids:
            enhanced = self.enhanced_inventory.get_module_enhanced(module_id)
            if enhanced:
                if enhanced.learning_index and enhanced.learning_index != 'not_assigned':
                    assigned_count += 1
                if enhanced.performance_index and enhanced.performance_index != 'not_assigned':
                    measured_count += 1
        
        # Calculate success probability
        assignment_rate = assigned_count / len(module_ids) if module_ids else 0
        measurement_rate = measured_count / len(module_ids) if module_ids else 0
        
        success_probability = (assignment_rate + measurement_rate) / 2
        
        return {
            'modules': module_ids,
            'assigned_modules': assigned_count,
            'measured_modules': measured_count,
            'assignment_rate': assignment_rate,
            'measurement_rate': measurement_rate,
            'predicted_success': success_probability,
            'confidence': min(1.0, len(module_ids) / 10)
        }
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def get_extended_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extended statistics"""
        return {
            'total_modules': len(self.enhanced_inventory.get_all_module_ids()),
            'superintelligences': len(self.enhanced_inventory.get_all_superintelligences()),
            'learning_coverage': self.get_learning_coverage(),
            'performance_coverage': self.get_performance_coverage(),
            'si_diversity': self.si_optimizer.si_stats,
            'learning_status': self.learning_optimizer.learning_status,
            'performance_status': self.performance_optimizer.performance_status
        }
    
    def print_extended_statistics(self):
        """Print extended statistics"""
        stats = self.get_extended_statistics()
        
        print("\n" + "="*80)
        print("DEEPALL INTEGRATION EXTENDED - STATISTICS")
        print("="*80)
        print(f"\nTotal Modules: {stats['total_modules']}")
        print(f"Superintelligences: {stats['superintelligences']}")
        print(f"Learning Coverage: {stats['learning_coverage']*100:.1f}%")
        print(f"Performance Coverage: {stats['performance_coverage']*100:.1f}%")
        
        print(f"\nSuperintelligence Stats:")
        for si, data in sorted(stats['si_diversity'].items())[:5]:
            print(f"  {si:15} - {data['module_count']:3} modules, diversity: {data['diversity']}")
        
        print("\n" + "="*80 + "\n")


# ============================================================================
# MAIN - TESTING
# ============================================================================

if __name__ == '__main__':
    import sys
    
    print("\n" + "="*80)
    print("DEEPALL INTEGRATION EXTENDED - TEST")
    print("="*80 + "\n")
    
    # Test 1: Initialize
    print("TEST 1: Initializing DeepALLIntegrationExtended...")
    try:
        inventory = EnhancedModuleInventory(
            'deepall_modules.json',
            'DeepALL_MASTER_V7_FINAL_WITH_ALL_REITERS.xlsx'
        )
        integration = DeepALLIntegrationExtended(inventory)
        print("  ✓ Initialized successfully\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        sys.exit(1)
    
    # Test 2: SI-based optimization
    print("TEST 2: SI-based optimization...")
    try:
        modules = integration.optimize_by_superintelligence(5)
        print(f"  ✓ Selected {len(modules)} modules: {modules}\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        sys.exit(1)
    
    # Test 3: Learning-aware optimization
    print("TEST 3: Learning-aware optimization...")
    try:
        modules = integration.optimize_for_learning(5)
        print(f"  ✓ Selected {len(modules)} modules")
        print(f"  ✓ Learning coverage: {integration.get_learning_coverage()*100:.1f}%\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        sys.exit(1)
    
    # Test 4: Performance-aware optimization
    print("TEST 4: Performance-aware optimization...")
    try:
        modules = integration.optimize_for_performance(5)
        print(f"  ✓ Selected {len(modules)} modules")
        print(f"  ✓ Performance coverage: {integration.get_performance_coverage()*100:.1f}%\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        sys.exit(1)
    
    # Test 5: Hybrid optimization
    print("TEST 5: Hybrid optimization...")
    try:
        modules = integration.optimize_hybrid(5)
        print(f"  ✓ Selected {len(modules)} modules: {modules}\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        sys.exit(1)
    
    # Test 6: Predictive analytics
    print("TEST 6: Predictive analytics...")
    try:
        prediction = integration.predict_module_synergy(modules)
        print(f"  ✓ Synergy prediction: {prediction['predicted_score']:.4f}")
        print(f"  ✓ Confidence: {prediction['confidence']:.4f}\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        sys.exit(1)
    
    # Print statistics
    integration.print_extended_statistics()
    
    print("✓ ALL TESTS PASSED\n")
