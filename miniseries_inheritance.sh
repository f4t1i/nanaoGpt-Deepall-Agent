#!/bin/bash

################################################################################
# Progressive Model Inheritance Training Pipeline
# Trains miniseries d10 → d11 → d12 → ... → d20
# Combines: ARS Optimizer + Fisher Information Regularization + Scaling Laws
#
# Author: Faton Duraku (2026)
# Usage: bash miniseries_inheritance.sh [--start 10] [--end 20] [--config config.yaml]
################################################################################

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

START_MODEL=10
END_MODEL=20
CONFIG_FILE="config.yaml"
DEVICE="cuda"
VERBOSE=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start)
            START_MODEL=$2
            shift 2
            ;;
        --end)
            END_MODEL=$2
            shift 2
            ;;
        --config)
            CONFIG_FILE=$2
            shift 2
            ;;
        --device)
            DEVICE=$2
            shift 2
            ;;
        --quiet)
            VERBOSE=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log() {
    if [ "$VERBOSE" = true ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    fi
}

log_section() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
}

log_error() {
    echo "[ERROR] $1" >&2
}

# ============================================================================
# VALIDATION
# ============================================================================

log_section "Validating Environment"

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python3 not found"
    exit 1
fi
log "✓ Python3 found: $(python3 --version)"

# Check required packages
log "Checking required packages..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python3 -c "import yaml; print('✓ PyYAML')"
python3 -c "import numpy; print('✓ NumPy')"
python3 -c "import matplotlib; print('✓ Matplotlib')"

# Check config file
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi
log "✓ Config file found: $CONFIG_FILE"

# Check required Python files
for file in regularization.py ars_optimizer.py train_miniseries.py evaluate.py utils.py; do
    if [ ! -f "$file" ]; then
        log_error "Required file not found: $file"
        exit 1
    fi
    log "✓ Found $file"
done

# ============================================================================
# SETUP
# ============================================================================

log_section "Setting Up Training Environment"

# Create directories
log "Creating directories..."
mkdir -p checkpoints weights fisher logs reports data tensorboard
log "✓ Directories created"

# Get device info
log "Detecting device..."
python3 << 'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Using CPU (training will be slow)")
EOF

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

log_section "Starting Miniseries Training: d${START_MODEL} → d${END_MODEL}"

log "Configuration:"
log "  Start Model: d${START_MODEL}"
log "  End Model: d${END_MODEL}"
log "  Config File: ${CONFIG_FILE}"
log "  Device: ${DEVICE}"

# Training loop
for model_idx in $(seq $START_MODEL $END_MODEL); do
    model_name="d${model_idx}"
    
    log_section "Training ${model_name}"
    
    # Create model-specific directories
    mkdir -p "checkpoints/${model_name}"
    mkdir -p "weights/${model_name}"
    mkdir -p "fisher/${model_name}"
    
    # Run training
    log "Starting training for ${model_name}..."
    python3 train_miniseries.py \
        --start "$model_idx" \
        --end "$model_idx" \
        --config "$CONFIG_FILE" \
        --device "$DEVICE" \
        --checkpoint-dir "checkpoints/${model_name}" \
        --weights-dir "weights/${model_name}" \
        --fisher-dir "fisher/${model_name}" \
        --log-file "logs/${model_name}.log"
    
    if [ $? -eq 0 ]; then
        log "✓ ${model_name} training completed successfully"
    else
        log_error "${model_name} training failed"
        exit 1
    fi
    
    # Compute Fisher Information for next model
    if [ "$model_idx" -lt "$END_MODEL" ]; then
        next_model="d$((model_idx + 1))"
        log "Computing Fisher Information for ${next_model}..."
        python3 << EOF
import torch
from regularization import FisherInformationMatrix
from train_miniseries import MiniseriesModel

# Load current model
model = MiniseriesModel()
model.load_state_dict(torch.load("weights/${model_name}/model.pt"))

# Compute Fisher Information
fisher = FisherInformationMatrix(model, 'cuda')
fisher.save("fisher/${model_name}/fisher_matrix.pt")
print(f"✓ Fisher Information saved for ${model_name}")
EOF
    fi
done

# ============================================================================
# EVALUATION
# ============================================================================

log_section "Evaluating Miniseries"

log "Computing metrics..."
python3 << 'EOF'
import json
from pathlib import Path
from evaluate import EvaluationReport, EvaluationMetrics

# Collect metrics from all models
metrics_list = []
for model_idx in range(10, 21):
    model_name = f"d{model_idx}"
    metrics_file = f"checkpoints/{model_name}/metrics.json"
    
    if Path(metrics_file).exists():
        with open(metrics_file) as f:
            metrics_data = json.load(f)
            metrics_list.append(EvaluationMetrics(**metrics_data))

# Generate report
if metrics_list:
    report_gen = EvaluationReport("./reports")
    report = report_gen.generate_metrics_report(metrics_list, "miniseries_report.json")
    report_gen.plot_scaling_curves(metrics_list, "scaling_curves.png")
    print(f"✓ Generated evaluation report for {len(metrics_list)} models")
else:
    print("No metrics found")
EOF

# ============================================================================
# SUMMARY
# ============================================================================

log_section "Training Complete"

log "Summary:"
log "  Models trained: d${START_MODEL} → d${END_MODEL}"
log "  Checkpoints: checkpoints/"
log "  Weights: weights/"
log "  Fisher Information: fisher/"
log "  Logs: logs/"
log "  Reports: reports/"

# Calculate total cost
log ""
log "Estimated Cost Breakdown:"
python3 << 'EOF'
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

scaling_laws = config.get('scaling_laws', {})
costs = scaling_laws.get('estimated_cost_usd', {})

total_cost = 0
for model, cost in costs.items():
    if model.startswith('d'):
        print(f"  {model}: ${cost:.2f}")
        total_cost += cost

print(f"\nTotal estimated cost: ${total_cost:.2f}")
EOF

log ""
log "✓ All training complete!"
log "Next steps:"
log "  1. Review logs in logs/"
log "  2. Check metrics in reports/"
log "  3. Evaluate scaling curves: reports/scaling_curves.png"
log "  4. Compare with GPT-2/GPT-3 baselines"

exit 0
