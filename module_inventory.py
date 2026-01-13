import json
from typing import List, Dict, Optional

class Module:
    def __init__(self, module_id: str, name: str, category: str, function: str, ai_training_method: str):
        self.id = module_id
        self.name = name
        self.category = str(category)
        self.function = str(function)
        self.ai_training_method = str(ai_training_method)
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "function": self.function,
            "ai_training_method": self.ai_training_method
        }

class ModuleInventory:
    def __init__(self, json_file: str):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.modules = {}
        for module_data in data:
            module = Module(
                str(module_data.get('id', '')),
                str(module_data.get('name', '')),
                str(module_data.get('category', '')),
                str(module_data.get('function', '')),
                str(module_data.get('ai_training_method', ''))
            )
            self.modules[module.id] = module
    
    def get_module(self, module_id: str) -> Optional[Module]:
        return self.modules.get(module_id)
    
    def get_all_module_ids(self) -> List[str]:
        return list(self.modules.keys())
    
    def get_modules_by_category(self, category: str) -> List[Module]:
        return [m for m in self.modules.values() if m.category == category]
    
    def get_modules_by_ai_method(self, method: str) -> List[Module]:
        return [m for m in self.modules.values() if m.ai_training_method == method]
    
    def print_statistics(self):
        categories = set(str(m.category) for m in self.modules.values())
        ai_methods = set(str(m.ai_training_method) for m in self.modules.values())
        print(f"✓ Loaded {len(self.modules)} modules from deepall_modules.json")
        print(f"Categories: {len(categories)} ({', '.join(sorted(categories))})")
        print(f"AI Methods: {len(ai_methods)} ({', '.join(sorted(ai_methods))})")

if __name__ == "__main__":
    inventory = ModuleInventory('deepall_modules.json')
    inventory.print_statistics()
