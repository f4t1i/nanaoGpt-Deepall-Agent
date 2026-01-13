# Research Findings: Progressive Inheritance Implementation

## Phase 1: Research Summary

### 1. Fisher Information Matrix (FIM) for Regularization

**Key Source:** Kirkpatrick et al. (2017) - "Overcoming Catastrophic Forgetting in Neural Networks" (EWC Paper)

#### What is Fisher Information?
- Measures parameter importance/sensitivity
- Diagonal approximation: F_i ≈ (∂L/∂w_i)²
- Computed from gradients on previous task data

#### EWC Regularization Formula
```
L_total = L_new + (λ/2) Σ_i F_i (w_i - w*_i)²

where:
  L_new = loss on new task
  F_i = Fisher Information diagonal
  w_i = current weights
  w*_i = previous task weights
  λ = regularization strength
```

#### Implementation Strategy
1. After training on task A, compute Fisher Information
2. When training on task B, add regularization term
3. Prevents large weight changes in important parameters
4. Diagonal approximation: O(n) instead of O(n²)

---

### 2. ARS Optimizer Components

#### Entropy Guard (Ψ_t)
- Detects periodicity in loss curve
- Uses Lag-1 autocorrelation: ρ = corr(L_t, L_{t-1})
- If |ρ| > threshold → training in loop → reduce learning rate
- Implementation: np.corrcoef(loss_history[-10:], range(10))

#### Surprise Gate (Φ_t)
- Detects unexpected gradient changes
- Surprise = |L_t - E[L_t]| / E[L_t]
- If surprise > threshold → dampen gradients
- Prevents divergence from sudden loss spikes

#### Chronos-Jitter (χ_t)
- Adds controlled noise to break periodicity
- Only active when Ψ_t < threshold
- Noise scale: σ = 0.01 × learning_rate
- Helps escape local minima

#### Combined Damping Factor
```
damping = Ψ_t × Φ_t × (1 + χ_t)
gradient *= damping
```

---

### 3. Progressive Inheritance Strategy

#### Model Sizes (d10-d20)
```
d10: 7M params
d11: 14M params (2× d10)
d12: 28M params (2× d11)
...
d20: 7.2B params

Each step: 2× parameters
```

#### Training Strategy
```
d10: Train from scratch (100 epochs)
d11: Load d10 weights, fine-tune (50 epochs) + Regularization
d12: Load d11 weights, fine-tune (50 epochs) + Regularization
...
d20: Load d19 weights, fine-tune (50 epochs) + Regularization
```

#### Cost Savings
```
Standard: 10 models × 2h = 20h
Progressive: 1×2h + 9×1h = 11h (45% reduction)
With ARS overhead: ~10% slower = 12h (40% reduction)
```

---

### 4. Scaling Laws (Karpathy's nanoGPT)

#### Chinchilla Ratio
```
Loss(N, D) ∝ N^(-0.5) + D^(-0.5)

Optimal ratio: D/N = 20 (Chinchilla)
nanoGPT ratio: D/N = 8

For fixed FLOPs:
  FLOPs = 6ND
  If D/N = 8: train 7B model on 56B tokens
```

#### Miniseries Validation
- Train d10-d20 with fixed FLOPs budget
- Plot loss vs. model size
- Should match Chinchilla scaling curve
- Validate with CORE score (from DCLM paper)

---

### 5. Implementation Priorities

#### High Priority (Core Functionality)
1. **regularization.py** - Fisher Information calculation
2. **ars_optimizer.py** - ARS gradient damping
3. **train_miniseries.py** - Main training loop

#### Medium Priority (Configuration & Utilities)
4. **config.yaml** - Hyperparameters
5. **utils.py** - Helper functions
6. **evaluate.py** - CORE score calculation

#### Low Priority (Automation)
7. **miniseries_inheritance.sh** - Bash script

---

## Phase 2: Implementation Checklist

### regularization.py
- [ ] Fisher Information diagonal computation
- [ ] Regularization loss function
- [ ] Integration with training loop
- [ ] Test on simple dataset

### ars_optimizer.py
- [ ] Entropy Guard implementation
- [ ] Surprise Gate implementation
- [ ] Chronos-Jitter implementation
- [ ] Combined damping factor
- [ ] Test stability improvement

### train_miniseries.py
- [ ] Model initialization (d10-d20)
- [ ] Data loading pipeline
- [ ] Training loop with ARS + Regularization
- [ ] Checkpoint saving
- [ ] Metrics tracking

### config.yaml
- [ ] Model sizes
- [ ] Training hyperparameters
- [ ] ARS parameters
- [ ] Regularization strength
- [ ] Data paths

### utils.py
- [ ] Data loading utilities
- [ ] Checkpoint management
- [ ] Metrics calculation
- [ ] Logging setup

### evaluate.py
- [ ] CORE score calculation
- [ ] Loss curve plotting
- [ ] Scaling law validation
- [ ] Comparison with GPT-2/3

### miniseries_inheritance.sh
- [ ] Environment setup
- [ ] Model training orchestration
- [ ] Checkpoint management
- [ ] Result aggregation

---

## Phase 3: Testing Strategy

### Unit Tests
- Fisher Information correctness
- ARS damping factor bounds
- Regularization loss magnitude
- Checkpoint save/load

### Integration Tests
- Full training loop (d10-d11)
- Progressive inheritance (d10→d11→d12)
- Metrics consistency
- Cost tracking

### Validation Tests
- Loss curve smoothness (ARS benefit)
- Scaling law adherence
- CORE score alignment
- Memory usage

---

## References

1. **EWC Paper:** Kirkpatrick, J., et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." PNAS.
   - URL: https://arxiv.org/pdf/1612.00796
   - Key: Fisher Information Matrix for continual learning

2. **Chinchilla Paper:** Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models."
   - Key: Scaling laws, optimal model/token ratio

3. **nanoGPT:** Karpathy, A. (2024). GitHub repository.
   - Key: Implementation of scaling laws, miniseries training

4. **Gradient Clipping:** Neptune.ai Blog (2025).
   - Key: Gradient damping techniques

5. **Catastrophic Forgetting:** IBM Think (2025).
   - Key: Overview of continual learning challenges
