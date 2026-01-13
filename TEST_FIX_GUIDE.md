# Test Implementation Fix Guide

## Current Status
- **Total Tests**: 21
- **Passing**: 13 (62%)
- **Failing**: 8 (38%)

## Failing Tests and Root Causes

### 1. TestRegularization.test_fisher_information_computation
**Error**: `AttributeError: 'FisherInformationMatrix' object has no attribute 'compute_fisher_information'`

**Root Cause**: Test calls `fisher.compute_fisher_information(loss)` but actual method is `fisher.compute_fisher(data_loader, num_batches, normalize)`

**Fix**: 
- Method requires a DataLoader, not just a loss value
- Need to create dummy DataLoader for testing
- Call: `fisher.compute_fisher(dummy_loader, num_batches=1)`

---

### 2. TestARSOptimizer.test_entropy_guard_detection
**Error**: `AttributeError: 'EntropyGuard' object has no attribute 'detect_periodicity'`

**Root Cause**: Test calls `guard.detect_periodicity(loss)` but actual method is `guard.compute_entropy()`

**Fix**:
- EntropyGuard has `update(loss)` to add to history, then `compute_entropy()` to get damping
- Correct flow:
  ```python
  for loss in [2.5, 2.4, 2.3, 2.4, 2.5]:
      guard.update(loss)
  entropy = guard.compute_entropy()
  ```

---

### 3. TestARSOptimizer.test_surprise_gate_damping
**Error**: `TypeError: SurpriseGate.compute_damping() takes 1 positional argument but 2 were given`

**Root Cause**: Test calls `gate.compute_damping(grad)` but method signature is `compute_damping(self)` with no grad parameter

**Fix**:
- SurpriseGate doesn't take gradients directly
- It computes damping from loss history
- Correct flow:
  ```python
  for loss in [2.5, 2.6, 2.4]:
      gate.update(loss)
  damping = gate.compute_damping()
  ```

---

### 4. TestARSOptimizer.test_chronos_jitter_generation
**Error**: `AttributeError: 'ChronosJitter' object has no attribute 'add_jitter'`

**Root Cause**: Test calls `jitter.add_jitter(x, entropy=0.3)` but actual method is `jitter.compute_jitter(learning_rate)`

**Fix**:
- ChronosJitter doesn't modify tensors directly
- It computes a scalar multiplier
- Correct flow:
  ```python
  jitter.set_entropy(0.5)  # Activate jitter
  multiplier = jitter.compute_jitter(learning_rate=1e-5)
  ```

---

### 5. TestTrainMiniseries.test_miniseries_trainer_training_step
**Error**: `AttributeError: 'MiniseriesTrainer' object has no attribute 'training_step'`

**Root Cause**: Test calls `trainer.training_step(batch)` but actual method is `trainer.train_epoch(train_loader, epoch, total_epochs)`

**Fix**:
- MiniseriesTrainer works with epochs, not individual batches
- Correct flow:
  ```python
  metrics = trainer.train_epoch(train_loader, epoch=0, total_epochs=5)
  ```

---

### 6. TestUtilities.test_learning_rate_scheduler
**Error**: `TypeError: LearningRateScheduler.__init__() got an unexpected keyword argument 'initial_lr'`

**Root Cause**: Test calls `LearningRateScheduler(initial_lr=1e-4, ...)` but actual signature is `__init__(self, optimizer, base_lr, strategy)`

**Fix**:
- LearningRateScheduler requires an optimizer instance
- Correct flow:
  ```python
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  scheduler = LearningRateScheduler(optimizer, base_lr=1e-4, strategy='constant')
  lr = scheduler.step()
  ```

---

### 7. TestUtilities.test_metrics_tracker
**Error**: `AttributeError: 'MetricsTracker' object has no attribute 'add_metric'`

**Root Cause**: Test calls `tracker.add_metric(name, value)` but actual method is `tracker.update(name, value)`

**Fix**:
- Correct method name is `update`, not `add_metric`
- Correct flow:
  ```python
  tracker = MetricsTracker()
  tracker.update('loss', 2.5)
  tracker.update('accuracy', 0.95)
  avg = tracker.get_average('loss')
  ```

---

### 8. TestIntegration.test_full_pipeline
**Error**: `AttributeError: 'MiniseriesTrainer' object has no attribute 'training_step'`

**Root Cause**: Same as #5 - uses non-existent `training_step` method

**Fix**: Use `train_epoch` instead

---

## Summary of Required Changes

| Test | Current Method | Actual Method | Parameter Changes |
|------|---|---|---|
| Fisher Info | `compute_fisher_information(loss)` | `compute_fisher(data_loader, num_batches, normalize)` | Needs DataLoader |
| Entropy Guard | `detect_periodicity(loss)` | `update(loss)` + `compute_entropy()` | Two-step process |
| Surprise Gate | `compute_damping(grad)` | `update(loss)` + `compute_damping()` | No grad parameter |
| Chronos Jitter | `add_jitter(x, entropy)` | `set_entropy(entropy)` + `compute_jitter(lr)` | Returns multiplier, not tensor |
| Trainer | `training_step(batch)` | `train_epoch(loader, epoch, total)` | Works with epochs |
| LR Scheduler | `LearningRateScheduler(initial_lr=...)` | `LearningRateScheduler(optimizer, base_lr, strategy)` | Needs optimizer |
| Metrics | `add_metric(name, value)` | `update(name, value)` | Method name change |

## Implementation Strategy

1. **Phase 1**: Fix individual component tests (Fisher, EntropyGuard, SurpriseGate, ChronosJitter)
2. **Phase 2**: Fix trainer tests (MiniseriesTrainer)
3. **Phase 3**: Fix utility tests (LearningRateScheduler, MetricsTracker)
4. **Phase 4**: Fix integration tests
5. **Phase 5**: Run full test suite and verify 21/21 passing

## Key Testing Principles

- **DataLoader Creation**: Use `torch.utils.data.DataLoader(TensorDataset(...))` for tests
- **Two-Step Process**: ARS components use `update()` then `compute_*()` pattern
- **Optimizer Dependency**: LearningRateScheduler needs real optimizer instance
- **Method Names**: Always check actual implementation before writing tests
