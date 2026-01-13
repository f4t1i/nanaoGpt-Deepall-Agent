#!/bin/bash

# ============================================================================
# RunPod Training Setup Script for DeepALL Agent
# Qwen3-VL-8B with Progressive Inheritance
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "DeepALL Agent - RunPod Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Step 1: System Update
# ============================================================================

echo -e "${BLUE}[Step 1] Updating system packages...${NC}"
apt update && apt upgrade -y
apt install -y git wget curl nano htop tmux build-essential

# ============================================================================
# Step 2: Python and PyTorch Installation
# ============================================================================

echo -e "${BLUE}[Step 2] Installing PyTorch with CUDA support...${NC}"
pip install --upgrade pip setuptools wheel

# PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
EOF

# ============================================================================
# Step 3: Install ML Dependencies
# ============================================================================

echo -e "${BLUE}[Step 3] Installing ML dependencies...${NC}"
pip install transformers==4.36.0
pip install datasets accelerate bitsandbytes peft
pip install pyyaml tqdm tensorboard wandb
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# ============================================================================
# Step 4: Clone DeepALL Agent Repository
# ============================================================================

echo -e "${BLUE}[Step 4] Cloning DeepALL Agent repository...${NC}"
cd ~
git clone https://github.com/f4t1i/nanoGpt-Deepall-Agent.git
cd nanoGpt-Deepall-Agent

# Install project requirements
pip install -r requirements.txt 2>/dev/null || echo "No requirements.txt found"

# ============================================================================
# Step 5: Create Directory Structure
# ============================================================================

echo -e "${BLUE}[Step 5] Creating directory structure...${NC}"
mkdir -p ~/data/training_data
mkdir -p ~/models/checkpoints
mkdir -p ~/models/weights
mkdir -p ~/models/fisher
mkdir -p ~/outputs/logs
mkdir -p ~/outputs/tensorboard

# ============================================================================
# Step 6: Hugging Face Login (Optional)
# ============================================================================

echo -e "${YELLOW}[Step 6] Hugging Face Authentication${NC}"
echo "If you have a Hugging Face account, you can login now."
echo "This is optional but recommended for downloading models."
echo ""
echo "Do you want to login to Hugging Face? (y/n)"
read -r hf_login

if [ "$hf_login" = "y" ] || [ "$hf_login" = "Y" ]; then
    huggingface-cli login
else
    echo "Skipping Hugging Face login"
fi

# ============================================================================
# Step 7: Download Qwen3-VL-8B Model (Optional)
# ============================================================================

echo -e "${YELLOW}[Step 7] Model Download${NC}"
echo "Do you want to download Qwen3-VL-8B model now? (y/n)"
echo "Note: This is ~16GB and will take a while."
read -r download_model

if [ "$download_model" = "y" ] || [ "$download_model" = "Y" ]; then
    echo -e "${BLUE}Downloading Qwen3-VL-8B...${NC}"
    python3 << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Downloading Qwen3-VL-8B model...")
model_name = "Qwen/Qwen3-VL-8B"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"✓ Tokenizer downloaded")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✓ Model downloaded")
    print(f"Model size: {model.num_parameters() / 1e9:.2f}B parameters")
except Exception as e:
    print(f"✗ Error downloading model: {e}")
    print("You can download it manually later using:")
    print(f"python3 -c \"from transformers import AutoModel; AutoModel.from_pretrained('{model_name}')\"")
EOF
else
    echo "Skipping model download. You can download it later."
fi

# ============================================================================
# Step 8: Verify Installation
# ============================================================================

echo -e "${BLUE}[Step 8] Verifying installation...${NC}"

python3 << 'EOF'
import sys
import torch
import transformers

print("\n" + "="*50)
print("INSTALLATION VERIFICATION")
print("="*50)

checks = {
    "Python": sys.version.split()[0],
    "PyTorch": torch.__version__,
    "Transformers": transformers.__version__,
    "CUDA": torch.cuda.is_available(),
    "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
}

for key, value in checks.items():
    status = "✓" if value else "✗"
    print(f"{status} {key}: {value}")

print("="*50 + "\n")
EOF

# ============================================================================
# Step 9: Create Training Script Wrapper
# ============================================================================

echo -e "${BLUE}[Step 9] Creating training script wrapper...${NC}"

cat > ~/start_training.sh << 'TRAIN_EOF'
#!/bin/bash

# Training wrapper script for RunPod

cd ~/nanoGpt-Deepall-Agent

echo "Starting DeepALL Agent Training..."
echo "Config: config_qwen3vl_runpod.yaml"
echo "Output: /root/outputs/logs/training.log"
echo ""

# Start training with nohup (continues even if SSH disconnects)
nohup python3 train_miniseries.py \
    --config config_qwen3vl_runpod.yaml \
    --output_dir /root/outputs \
    --log_dir /root/outputs/logs \
    > /root/outputs/logs/training.log 2>&1 &

echo "Training started in background (PID: $!)"
echo "Monitor with: tail -f /root/outputs/logs/training.log"
TRAIN_EOF

chmod +x ~/start_training.sh

# ============================================================================
# Step 10: Summary
# ============================================================================

echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Upload your training data to: /root/data/training_data/"
echo "   scp -P <port> -r /path/to/data/* root@<pod-ip>:/root/data/training_data/"
echo ""
echo "2. Start training:"
echo "   bash ~/start_training.sh"
echo ""
echo "3. Monitor training:"
echo "   tail -f /root/outputs/logs/training.log"
echo ""
echo "4. Check GPU usage:"
echo "   nvidia-smi -l 1"
echo ""
echo "5. Download results:"
echo "   scp -P <port> -r root@<pod-ip>:/root/models/checkpoints ~/Downloads/"
echo ""
echo -e "${GREEN}Ready for training!${NC}"
