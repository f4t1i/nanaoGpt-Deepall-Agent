import json
from typing import List, Dict, Set, Tuple
from module_inventory import ModuleInventory, Module

class DeepALLIntegration:
    """Synergy detection and optimization across DeepALL modules"""
    
    def __init__(self, inventory: ModuleInventory):
        self.inventory = inventory
        self.synergy_cache = {}
        self.module_relationships = {}
    
    def detect_synergies(self, module_ids: List[str]) -> Dict[str, any]:
        """Detect synergies between selected modules"""
        modules = [self.inventory.get_module(mid) for mid in module_ids if self.inventory.get_module(mid)]
        
        if not modules:
            return {"synergies": [], "total_score": 0.0}
        
        synergies = []
        
        # Check category synergies
        categories = set(m.category for m in modules)
        category_synergy = self._calculate_category_synergy(categories)
        if category_synergy > 0:
            synergies.append({
                "type": "category",
                "categories": list(categories),
                "score": round(category_synergy, 4)
            })
        
        # Check AI method synergies
        ai_methods = set(m.ai_training_method for m in modules)
        method_synergy = self._calculate_method_synergy(ai_methods)
        if method_synergy > 0:
            synergies.append({
                "type": "ai_method",
                "methods": list(ai_methods),
                "score": round(method_synergy, 4)
            })
        
        # Check pairwise synergies
        pairwise_synergies = self._calculate_pairwise_synergies(modules)
        synergies.extend(pairwise_synergies)
        
        total_score = sum(s['score'] for s in synergies)
        
        return {
            "synergies": synergies,
            "total_score": round(total_score, 4),
            "num_modules": len(modules)
        }
    
    def _calculate_category_synergy(self, categories: Set[str]) -> float:
        """Calculate synergy from category diversity"""
        # More categories = higher synergy (up to a point)
        num_categories = len(categories)
        if num_categories == 1:
            return 0.0
        elif num_categories == 2:
            return 0.3
        elif num_categories == 3:
            return 0.5
        else:
            return 0.6
    
    def _calculate_method_synergy(self, methods: Set[str]) -> float:
        """Calculate synergy from AI method diversity"""
        num_methods = len(methods)
        if num_methods == 1:
            return 0.0
        elif num_methods == 2:
            return 0.25
        else:
            return 0.4
    
    def _calculate_pairwise_synergies(self, modules: List[Module]) -> List[Dict]:
        """Calculate pairwise synergies between modules"""
        pairwise = []
        
        for i in range(len(modules)):
            for j in range(i + 1, len(modules)):
                m1, m2 = modules[i], modules[j]
                
                # Calculate compatibility
                compatibility = 0.0
                
                # Same category bonus
                if m1.category == m2.category:
                    compatibility += 0.2
                
                # Different AI methods bonus
                if m1.ai_training_method != m2.ai_training_method:
                    compatibility += 0.15
                
                # Random additional synergy
                compatibility += 0.1
                
                if compatibility > 0:
                    pairwise.append({
                        "type": "pairwise",
                        "module_1": m1.id,
                        "module_2": m2.id,
                        "score": round(compatibility, 4)
                    })
        
        return pairwise
    
    def optimize_module_selection(self, num_modules: int = 5) -> Dict:
        """Recommend optimal module combinations"""
        all_modules = self.inventory.get_all_module_ids()
        
        if num_modules > len(all_modules):
            num_modules = len(all_modules)
        
        # Try multiple combinations and find best
        best_combination = None
        best_score = -1.0
        
        for _ in range(100):  # Try 100 random combinations
            import random
            combination = random.sample(all_modules, num_modules)
            synergy_result = self.detect_synergies(combination)
            total_score = synergy_result['total_score']
            
            if total_score > best_score:
                best_score = total_score
                best_combination = combination
        
        return {
            "recommended_modules": best_combination,
            "synergy_analysis": self.detect_synergies(best_combination),
            "optimization_score": round(best_score, 4)
        }
    
    def get_module_relationships(self, module_id: str) -> Dict:
        """Get relationships and synergies for a specific module"""
        module = self.inventory.get_module(module_id)
        
        if not module:
            return {"error": f"Module {module_id} not found"}
        
        # Find related modules
        related_by_category = self.inventory.get_modules_by_category(module.category)
        related_by_method = self.inventory.get_modules_by_ai_method(module.ai_training_method)
        
        return {
            "module_id": module_id,
            "module_name": module.name,
            "category": module.category,
            "ai_method": module.ai_training_method,
            "related_by_category": [m.id for m in related_by_category if m.id != module_id],
            "related_by_method": [m.id for m in related_by_method if m.id != module_id],
            "num_related_by_category": len([m for m in related_by_category if m.id != module_id]),
            "num_related_by_method": len([m for m in related_by_method if m.id != module_id])
        }
    
    def export_synergy_analysis(self, module_ids: List[str], filename: str):
        """Export synergy analysis to JSON"""
        analysis = self.detect_synergies(module_ids)
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✓ Exported synergy analysis to {filename}")

if __name__ == "__main__":
    inventory = ModuleInventory('deepall_modules.json')
    integration = DeepALLIntegration(inventory)
    
    # Detect synergies
    import random
    sample_modules = random.sample(inventory.get_all_module_ids(), 3)
    synergies = integration.detect_synergies(sample_modules)
    print(f"✓ Synergy Detection - Score: {synergies['total_score']:.4f}")
    
    # Optimize selection
    optimal = integration.optimize_module_selection(num_modules=5)
    print(f"✓ Optimization - Best Score: {optimal['optimization_score']:.4f}")
    
    # Get relationships
    if sample_modules:
        relationships = integration.get_module_relationships(sample_modules[0])
        print(f"✓ Relationships - Related modules: {relationships['num_related_by_category']}")
    
    # Export
    integration.export_synergy_analysis(sample_modules, 'synergy_analysis.json')
