# RunPod A100 Deployment Guide - DeepALL Agent

**Ziel**: Qwen3-VL-8B mit Progressive Inheritance auf RunPod A100 trainieren  
**Geschätzte Kosten**: $4-6 für vollständiges Training  
**Geschätzte Zeit**: 10-12 Stunden  
**Status**: Bereit zum Deployment

---

## Phase 1: RunPod Pod Deployment

### Schritt 1.1: Pod-Typ auswählen

1. Gehen Sie auf https://www.runpod.io
2. Klicken Sie auf **"Deploy a Pod"** (rosa Button)
3. Wählen Sie **"GPU Cloud"** (nicht Serverless)
4. Suchen Sie nach **"A100"** (80GB VRAM)

### Schritt 1.2: Pod-Konfiguration

**Wichtige Einstellungen:**

| Parameter | Wert | Grund |
|-----------|------|-------|
| **GPU** | NVIDIA A100 (80GB) | Für Qwen3-VL-8B Training |
| **vCPU** | 8-16 | Datenverarbeitung |
| **RAM** | 32-64 GB | Speicher für Modell + Daten |
| **Storage** | 100-200 GB | Modelle + Checkpoints |
| **Template** | PyTorch 2.0+ | Für Training |

### Schritt 1.3: Pod starten

1. Klicken Sie **"Deploy"**
2. Warten Sie, bis Pod lädt (2-5 Minuten)
3. Sie sehen dann: **"Pod is running"**
4. Notieren Sie sich die **Pod-ID** (z.B. `abc123xyz`)

### Schritt 1.4: SSH-Zugang einrichten

1. Klicken Sie auf **"Connect"** im Pod-Dashboard
2. Sie sehen SSH-Befehl: `ssh root@<pod-ip> -p <port>`
3. Kopieren Sie den kompletten Befehl
4. Öffnen Sie Terminal/PowerShell auf Ihrem Computer
5. Führen Sie SSH-Befehl aus

**Beispiel:**
```bash
ssh root@123.45.67.89 -p 22
```

---

## Phase 2: Umgebung auf RunPod einrichten

### Schritt 2.1: Grundlegende Pakete installieren

Nachdem Sie SSH-Zugang haben:

```bash
# Update system
apt update && apt upgrade -y

# Install essentials
apt install -y git wget curl nano python3-pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and dependencies
pip install transformers datasets accelerate bitsandbytes peft

# Install project dependencies
pip install pyyaml tqdm tensorboard wandb
```

### Schritt 2.2: DeepALL Agent Code klonen

```bash
# Navigate to home directory
cd ~

# Clone the repository
git clone https://github.com/f4t1i/nanoGpt-Deepall-Agent.git
cd nanoGpt-Deepall-Agent

# Install project requirements
pip install -r requirements.txt
```

### Schritt 2.3: Datenverzeichnis vorbereiten

```bash
# Create data directory
mkdir -p ~/data/training_data
mkdir -p ~/models/checkpoints
mkdir -p ~/outputs/logs

# Verify structure
ls -la ~/
```

---

## Phase 3: Konfiguration für Qwen3-VL-8B

### Schritt 3.1: config.yaml aktualisieren

Erstellen Sie `config.yaml` mit folgendem Inhalt:

```yaml
# DeepALL Agent Configuration for Qwen3-VL-8B
# RunPod A100 Deployment

model:
  name: "Qwen/Qwen3-VL-8B"
  type: "vision-language"
  vocab_size: 152064
  hidden_size: 3584
  num_layers: 32
  num_heads: 32
  head_dim: 112
  context_length: 256000
  vision_enabled: true
  
  # Vision-specific config
  vision:
    image_size: 1024
    patch_size: 14
    num_vision_layers: 32
    vision_hidden_size: 3584

training:
  batch_size: 2  # Per GPU (A100 can handle more)
  gradient_accumulation_steps: 4
  learning_rate: 1e-4
  warmup_steps: 500
  max_steps: 10000
  eval_steps: 500
  save_steps: 1000
  
  # ARS Optimizer settings
  ars_enabled: true
  entropy_threshold: 0.5
  surprise_damping_range: [0.5, 1.0]
  chronos_jitter_range: [0.8, 1.2]
  
  # Progressive Inheritance
  progressive_inheritance:
    enabled: true
    start_depth: 10
    end_depth: 18
    fisher_weight: 0.1
    gamma: 0.01

optimizer:
  type: "ARSAdam"
  beta1: 0.9
  beta2: 0.999
  epsilon: 1e-8
  weight_decay: 0.01

regularization:
  fisher_information:
    enabled: true
    num_batches: 100
    normalize: true
  progressive_inheritance:
    enabled: true
    gamma: 0.01

hardware:
  device: "cuda"
  mixed_precision: "fp16"
  gradient_checkpointing: true
  max_memory_percentage: 0.9

paths:
  data_dir: "/root/data/training_data"
  model_cache: "/root/models"
  checkpoint_dir: "/root/models/checkpoints"
  output_dir: "/root/outputs"
  log_dir: "/root/outputs/logs"

logging:
  level: "INFO"
  log_file: "/root/outputs/logs/training.log"
  wandb_enabled: false  # Set to true if you want W&B tracking
```

### Schritt 3.2: config.yaml auf RunPod hochladen

```bash
# Auf Ihrem lokalen Computer:
scp -P <port> config.yaml root@<pod-ip>:/root/nanoGpt-Deepall-Agent/

# Oder auf RunPod:
nano config.yaml
# (Paste content, Ctrl+X, Y, Enter)
```

---

## Phase 4: Datendateien hochladen

### Schritt 4.1: Daten vorbereiten

Sie haben 29 Datendateien. Diese müssen auf RunPod hochgeladen werden.

**Option A: Mit SCP (schneller)**

```bash
# Auf Ihrem lokalen Computer:
scp -P <port> -r /path/to/your/data/* root@<pod-ip>:/root/data/training_data/

# Beispiel:
scp -P 22 -r ~/Documents/training_data/* root@123.45.67.89:/root/data/training_data/
```

**Option B: Mit rsync (mit Fortschritt)**

```bash
rsync -avz -e "ssh -p <port>" /path/to/your/data/ root@<pod-ip>:/root/data/training_data/
```

### Schritt 4.2: Daten verifizieren

```bash
# Auf RunPod (SSH):
ls -lh /root/data/training_data/
wc -l /root/data/training_data/*  # Zeilenanzahl
du -sh /root/data/training_data/  # Gesamtgröße
```

---

## Phase 5: Training starten

### Schritt 5.1: Training-Script ausführen

```bash
# Auf RunPod (SSH):
cd /root/nanoGpt-Deepall-Agent

# Starten Sie das Training
python3 train_miniseries.py \
  --config config.yaml \
  --output_dir /root/outputs \
  --log_dir /root/outputs/logs

# Oder mit dem Shell-Script:
bash miniseries_inheritance.sh
```

### Schritt 5.2: Training überwachen

```bash
# In separatem SSH-Terminal:
tail -f /root/outputs/logs/training.log

# Oder mit watch:
watch -n 5 'tail -20 /root/outputs/logs/training.log'

# GPU-Auslastung prüfen:
nvidia-smi -l 1  # Aktualisiert jede Sekunde
```

---

## Phase 6: Ergebnisse herunterladen

### Schritt 6.1: Modelle und Checkpoints herunterladen

```bash
# Auf Ihrem lokalen Computer:
scp -P <port> -r root@<pod-ip>:/root/models/checkpoints ~/Downloads/deepall_checkpoints/
scp -P <port> -r root@<pod-ip>:/root/outputs ~/Downloads/deepall_outputs/
```

### Schritt 6.2: Logs und Metriken

```bash
# Logs herunterladen
scp -P <port> -r root@<pod-ip>:/root/outputs/logs ~/Downloads/deepall_logs/

# Metriken anschauen
cat ~/Downloads/deepall_logs/training.log | grep "Epoch\|Loss\|Accuracy"
```

---

## Troubleshooting

### Problem: "CUDA out of memory"

**Lösung:**
```yaml
# In config.yaml:
training:
  batch_size: 1  # Reduzieren
  gradient_accumulation_steps: 8  # Erhöhen
```

### Problem: "Pod disconnected"

**Lösung:**
```bash
# SSH-Verbindung mit screen/tmux halten
screen -S training
# Starten Sie Training
python3 train_miniseries.py ...
# Detach: Ctrl+A, D
# Reattach: screen -r training
```

### Problem: "Modell lädt nicht"

**Lösung:**
```bash
# Hugging Face Token einrichten
huggingface-cli login
# Geben Sie Ihren HF Token ein

# Oder in config:
export HF_TOKEN="your_token_here"
```

---

## Kostenoptimierung

| Aktion | Kosten-Einsparung |
|--------|------------------|
| Spot-Instanzen verwenden | -70% |
| Kleinere Batch-Size | -20% |
| Weniger Checkpoints speichern | -15% |
| Kürzeres Training (weniger Epochs) | -30% |

**Empfehlung**: Verwenden Sie **Spot-Instanzen** für Training (können unterbrochen werden, aber billiger)

---

## Geschätzte Kosten & Zeit

| Phase | Zeit | Kosten |
|-------|------|--------|
| Pod Setup | 5 min | $0 |
| Daten Upload | 30 min | $0 |
| Training d10→d18 | 10 Stunden | $4-6 |
| Ergebnisse Download | 30 min | $0 |
| **TOTAL** | **~11 Stunden** | **~$4-6** |

---

## Nächste Schritte

1. ✅ Pod deployen (Schritt 1)
2. ✅ Umgebung einrichten (Schritt 2)
3. ✅ config.yaml hochladen (Schritt 3)
4. ✅ Daten hochladen (Schritt 4)
5. ✅ Training starten (Schritt 5)
6. ✅ Ergebnisse herunterladen (Schritt 6)

---

## Support & Ressourcen

- **RunPod Docs**: https://docs.runpod.io
- **Qwen3-VL Docs**: https://huggingface.co/Qwen/Qwen3-VL-8B
- **DeepALL Agent Repo**: https://github.com/f4t1i/nanoGpt-Deepall-Agent

---

**Status**: ✅ Bereit zum Deployment  
**Letzte Aktualisierung**: 2026-01-13  
**Version**: 1.0
