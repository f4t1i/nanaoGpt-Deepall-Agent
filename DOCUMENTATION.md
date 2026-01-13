# nanoGPT-DeepALL-Agent Framework

Complete framework for training language models to intelligently select and orchestrate DeepALL modules.

## Features

- **219 DeepALL Modules**: Real-world module inventory
- **4 Training Methods**: SFT, RL, ICL, Continuous Learning
- **Synergy Detection**: Identify optimal module combinations
- **Reward System**: Multi-component reward calculation
- **Training Orchestrator**: Unified interface

## Quick Start

```python
from module_inventory import ModuleInventory
from training_orchestrator import TrainingOrchestrator

inventory = ModuleInventory('deepall_modules.json')
orchestrator = TrainingOrchestrator(inventory)
orchestrator.run_all_training(sft_samples=50, rl_episodes=5)
```

## Documentation

- See DOCUMENTATION.md for detailed information
- See QUICKSTART.md for getting started
- See ROADMAP.md for future plans
- See ARS_INTEGRATION_ROADMAP.md for Phase 2

## License

MIT
