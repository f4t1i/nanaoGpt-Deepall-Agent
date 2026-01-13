from dataclasses import dataclass
from typing import List

@dataclass
class ExecutionResult:
    task_id: str
    predicted_modules: List[str]
    actual_modules: List[str]
    execution_time: float
    resource_usage: float
    success: bool

class RewardSystem:
    def __init__(self, selection_weight=0.5, sequence_weight=0.3, efficiency_weight=0.2):
        self.selection_weight = selection_weight
        self.sequence_weight = sequence_weight
        self.efficiency_weight = efficiency_weight
    
    def calculate_reward(self, result: ExecutionResult) -> float:
        if not result.success:
            return -0.578
        
        selection_accuracy = len(set(result.predicted_modules) & set(result.actual_modules)) / len(set(result.actual_modules))
        
        sequence_correctness = 1.0 if result.predicted_modules == result.actual_modules else 0.5
        
        efficiency_score = max(0, 1.0 - (result.execution_time / 10.0 + result.resource_usage))
        
        reward = (self.selection_weight * selection_accuracy + 
                 self.sequence_weight * sequence_correctness + 
                 self.efficiency_weight * efficiency_score)
        
        return reward

if __name__ == "__main__":
    reward_system = RewardSystem()
    result = ExecutionResult(
        task_id="task_001",
        predicted_modules=["m001", "m002"],
        actual_modules=["m001", "m002"],
        execution_time=5.0,
        resource_usage=0.3,
        success=True
    )
    reward = reward_system.calculate_reward(result)
    print(f"✓ Reward system test: {reward}")
