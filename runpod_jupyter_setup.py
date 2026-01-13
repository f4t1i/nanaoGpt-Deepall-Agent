#!/usr/bin/env python3
"""
Complete RunPod A100 Setup Script for Jupyter
No bash, pure Python!
"""

import subprocess
import os
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run command and print output"""
    print(f"\n{'='*80}")
    if description:
        print(f"📌 {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.returncode}")
        return False
    else:
        print(f"✅ Success!")
        return True

# ============================================================================
# STEP 1: CLONE REPOSITORY
# ============================================================================
print("\n" + "="*80)
print("STEP 1: CLONE DEEPALL AGENT REPOSITORY")
print("="*80)

repo_url = "https://github.com/f4t1i/nanoGpt-Deepall-Agent.git"
repo_dir = Path("/root/nanoGpt-Deepall-Agent")

if repo_dir.exists():
    print(f"✓ Repository already exists at {repo_dir}")
    print("Pulling latest changes...")
    os.chdir(repo_dir)
    subprocess.run("git pull", shell=True)
else:
    print(f"Cloning {repo_url}...")
    subprocess.run(f"git clone {repo_url} {repo_dir}", shell=True)
    os.chdir(repo_dir)

print(f"✓ Repository ready at {repo_dir}")

# ============================================================================
# STEP 2: CREATE DIRECTORIES
# ============================================================================
print("\n" + "="*80)
print("STEP 2: CREATE DIRECTORIES")
print("="*80)

dirs = [
    "/root/data",
    "/root/nanoGpt-Deepall-Agent/data",
    "/root/checkpoints",
    "/root/weights",
    "/root/fisher",
    "/root/logs",
    "/root/results"
]

for dir_path in dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✓ {dir_path}")

# ============================================================================
# STEP 3: UPGRADE PIP
# ============================================================================
print("\n" + "="*80)
print("STEP 3: UPGRADE PIP")
print("="*80)

subprocess.run(f"{sys.executable} -m pip install --upgrade pip setuptools wheel", shell=True)
print("✓ pip upgraded")

# ============================================================================
# STEP 4: INSTALL PYTORCH
# ============================================================================
print("\n" + "="*80)
print("STEP 4: INSTALL PYTORCH WITH CUDA")
print("="*80)

pytorch_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
subprocess.run(pytorch_cmd, shell=True)
print("✓ PyTorch installed")

# ============================================================================
# STEP 5: INSTALL ML DEPENDENCIES
# ============================================================================
print("\n" + "="*80)
print("STEP 5: INSTALL ML DEPENDENCIES")
print("="*80)

packages = [
    "transformers==4.36.0",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "peft",
    "pyyaml",
    "tqdm",
    "tensorboard",
    "wandb",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pillow",
    "requests",
    "huggingface-hub",
    "psutil"
]

for package in packages:
    print(f"Installing {package}...")
    subprocess.run(f"{sys.executable} -m pip install {package}", shell=True, capture_output=True)

print(f"✓ All {len(packages)} packages installed")

# ============================================================================
# STEP 6: VERIFY INSTALLATION
# ============================================================================
print("\n" + "="*80)
print("STEP 6: VERIFY INSTALLATION")
print("="*80)

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except ImportError as e:
    print(f"❌ Error: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Error: {e}")

# ============================================================================
# STEP 7: LIST REPOSITORY FILES
# ============================================================================
print("\n" + "="*80)
print("STEP 7: REPOSITORY FILES")
print("="*80)

repo_files = list(Path(repo_dir).glob("*.py"))[:10]
for file in repo_files:
    print(f"  ✓ {file.name}")

print(f"\n✓ Total Python files: {len(list(Path(repo_dir).glob('*.py')))}")

# ============================================================================
# DONE
# ============================================================================
print("\n" + "="*80)
print("✅ SETUP COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Run: python3 download_model.py")
print("2. Run: python3 upload_data.py")
print("3. Run: python3 start_training.py")
print("\n" + "="*80)
