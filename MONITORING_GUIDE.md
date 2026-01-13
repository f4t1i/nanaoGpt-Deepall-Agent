# RunPod Training Monitoring & Troubleshooting Guide

**Ziel**: Training überwachen und Probleme beheben  
**Tools**: SSH, nvidia-smi, tail, tmux

---

## Teil 1: Training überwachen

### 1.1: Logs in Echtzeit anschauen

```bash
# SSH in Pod
ssh root@123.45.67.89 -p 22

# Logs live anschauen (letzte 20 Zeilen, aktualisiert sich)
tail -f /root/outputs/logs/training.log

# Nur neue Zeilen sehen (seit letzter Abfrage)
tail -f /root/outputs/logs/training.log | grep -E "Epoch|Loss|Accuracy"

# Letzte 100 Zeilen anschauen
tail -100 /root/outputs/logs/training.log

# Bestimmte Zeile suchen
grep "Error" /root/outputs/logs/training.log
```

### 1.2: GPU-Auslastung überwachen

```bash
# SSH in Pod
ssh root@123.45.67.89 -p 22

# GPU-Status (aktualisiert sich jede Sekunde)
nvidia-smi -l 1

# Nur GPU-Memory anschauen
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader -l 1

# Detaillierte GPU-Info
nvidia-smi -i 0 -q

# GPU-Prozesse
nvidia-smi pmon -c 1
```

### 1.3: System-Ressourcen überwachen

```bash
# SSH in Pod
ssh root@123.45.67.89 -p 22

# CPU und RAM (aktualisiert sich)
watch -n 1 'top -bn1 | head -20'

# Oder mit htop (schöner)
htop

# Speicherplatz prüfen
df -h /root/

# Disk-Nutzung
du -sh /root/models /root/outputs /root/data
```

### 1.4: Training-Prozess prüfen

```bash
# SSH in Pod
ssh root@123.45.67.89 -p 22

# Python-Prozess finden
ps aux | grep python

# Oder mit pgrep
pgrep -f train_miniseries

# Prozess-Details
ps aux | grep train_miniseries | grep -v grep
```

---

## Teil 2: Mit tmux arbeiten (empfohlen)

### 2.1: tmux Session starten

```bash
# SSH in Pod
ssh root@123.45.67.89 -p 22

# Neue tmux Session erstellen
tmux new-session -s training

# Training starten
cd ~/nanoGpt-Deepall-Agent
python3 train_miniseries.py --config config_qwen3vl_runpod.yaml

# Detach (Ctrl+B, D)
# Training läuft weiter im Hintergrund!

# Später reattach
tmux attach-session -t training

# Session beenden
tmux kill-session -t training
```

### 2.2: Mehrere tmux Windows

```bash
# In tmux Session:
# Neues Window: Ctrl+B, C
# Window wechseln: Ctrl+B, N (next) oder P (previous)
# Window liste: Ctrl+B, W

# Beispiel Setup:
# Window 1: Training
# Window 2: GPU-Monitoring (nvidia-smi)
# Window 3: Logs (tail -f)
```

---

## Teil 3: Metriken und Checkpoints

### 3.1: Checkpoints anschauen

```bash
# SSH in Pod
ssh root@123.45.67.89 -p 22

# Alle Checkpoints
ls -lh /root/models/checkpoints/

# Checkpoint-Größe
du -sh /root/models/checkpoints/*

# Neueste Checkpoints
ls -lht /root/models/checkpoints/ | head -10

# Checkpoint-Info (falls JSON)
cat /root/models/checkpoints/checkpoint-*/config.json | head -50
```

### 3.2: Training-Metriken

```bash
# SSH in Pod
ssh root@123.45.67.89 -p 22

# Logs filtern nach wichtigen Metriken
grep "Epoch" /root/outputs/logs/training.log

# Loss-Werte extrahieren
grep "loss" /root/outputs/logs/training.log | tail -20

# Durchschnittliche Loss pro Epoch
grep "Epoch" /root/outputs/logs/training.log | awk '{print $NF}'
```

### 3.3: TensorBoard (optional)

```bash
# SSH in Pod
ssh root@123.45.67.89 -p 22

# TensorBoard starten (im Hintergrund)
nohup tensorboard --logdir=/root/outputs/tensorboard --port=6006 &

# Auf Ihrem Computer: Port forwarding
ssh -L 6006:localhost:6006 root@123.45.67.89 -p 22

# Browser: http://localhost:6006
```

---

## Teil 4: Troubleshooting

### Problem 1: "CUDA out of memory"

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Lösung:**
```yaml
# In config_qwen3vl_runpod.yaml:
training:
  batch_size: 1  # Reduzieren von 2 zu 1
  gradient_accumulation_steps: 8  # Erhöhen von 4 zu 8
```

**Oder:**
```bash
# Training stoppen und neu starten mit kleinerer Batch-Size
# Ctrl+C (im tmux)
# Bearbeiten Sie config_qwen3vl_runpod.yaml
# Starten Sie neu
```

### Problem 2: "Training is very slow"

**Ursachen:**
- Langsame Daten-Laden
- Zu viele DataLoader-Worker
- Nicht genug GPU-Memory

**Lösungen:**
```yaml
# In config_qwen3vl_runpod.yaml:
data:
  num_workers: 8  # Erhöhen (default: 4)
  pin_memory: true  # Sicherstellen dass true
  prefetch_factor: 4  # Erhöhen (default: 2)

device:
  gradient_checkpointing: false  # Wenn nicht nötig, ausschalten
  flash_attention: true  # Einschalten
```

### Problem 3: "Pod disconnected / SSH timeout"

**Lösung:**
```bash
# Training mit nohup starten (läuft auch wenn SSH abbricht)
ssh root@123.45.67.89 -p 22 << 'EOF'
cd ~/nanoGpt-Deepall-Agent
nohup python3 train_miniseries.py \
    --config config_qwen3vl_runpod.yaml \
    > /root/outputs/logs/training.log 2>&1 &
echo $! > /tmp/training.pid
EOF

# Später wieder verbinden und Status prüfen
ssh root@123.45.67.89 -p 22 "tail -f /root/outputs/logs/training.log"
```

### Problem 4: "Model loading fails"

**Symptom:**
```
OSError: Can't load model. Model not found on huggingface.co
```

**Lösung:**
```bash
# SSH in Pod
ssh root@123.45.67.89 -p 22

# Hugging Face login
huggingface-cli login
# Geben Sie Ihren HF Token ein

# Oder Token setzen
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# Training erneut starten
```

### Problem 5: "Disk space full"

**Symptom:**
```
OSError: [Errno 28] No space left on device
```

**Lösung:**
```bash
# SSH in Pod
ssh root@123.45.67.89 -p 22

# Speicher prüfen
df -h /root/

# Alte Checkpoints löschen
rm -rf /root/models/checkpoints/checkpoint-[0-9]*

# Oder nur alte behalten
ls -t /root/models/checkpoints/ | tail -n +4 | xargs -I {} rm -rf /root/models/checkpoints/{}

# Logs komprimieren
gzip /root/outputs/logs/*.log
```

### Problem 6: "Training crashed / Process died"

**Lösung:**
```bash
# SSH in Pod
ssh root@123.45.67.89 -p 22

# Letzte Fehler im Log anschauen
tail -100 /root/outputs/logs/training.log | grep -i "error\|exception\|traceback"

# Oder ganzen Error-Stack
grep -A 20 "Traceback" /root/outputs/logs/training.log | tail -30

# Training von letztem Checkpoint fortsetzen
# (falls resume_from_checkpoint in config)
```

---

## Teil 5: Performance-Optimierung

### 5.1: Aktuelle Performance prüfen

```bash
# SSH in Pod
ssh root@123.45.67.89 -p 22

# Durchsatz (samples pro Sekunde)
grep "samples/sec" /root/outputs/logs/training.log | tail -5

# Durchschnittliche Batch-Zeit
grep "batch" /root/outputs/logs/training.log | tail -10

# GPU-Auslastung während Training
nvidia-smi dmon -c 10  # 10 Samples
```

### 5.2: Optimierungen

**Für schnelleres Training:**
```yaml
device:
  flash_attention: true  # Schneller
  gradient_checkpointing: false  # Wenn Speicher OK

training:
  batch_size: 4  # Erhöhen wenn möglich
  gradient_accumulation_steps: 2  # Reduzieren
```

**Für weniger Speicher:**
```yaml
device:
  gradient_checkpointing: true  # Speichern
  flash_attention: true  # Trotzdem schnell

training:
  batch_size: 1  # Reduzieren
  gradient_accumulation_steps: 8  # Erhöhen
```

---

## Teil 6: Ergebnisse herunterladen

### 6.1: Während Training läuft

```bash
# Auf Ihrem Computer:
# Checkpoints herunterladen (während Training läuft)
rsync -avz -e "ssh -p 22" \
    root@123.45.67.89:/root/models/checkpoints/ \
    ~/Downloads/deepall_checkpoints/

# Logs herunterladen
rsync -avz -e "ssh -p 22" \
    root@123.45.67.89:/root/outputs/logs/ \
    ~/Downloads/deepall_logs/
```

### 6.2: Nach Training

```bash
# Alles herunterladen
rsync -avz -e "ssh -p 22" \
    root@123.45.67.89:/root/models/ \
    ~/Downloads/deepall_models/

rsync -avz -e "ssh -p 22" \
    root@123.45.67.89:/root/outputs/ \
    ~/Downloads/deepall_outputs/
```

---

## Checkliste für erfolgreiche Überwachung

- [ ] SSH-Verbindung funktioniert
- [ ] Training läuft (ps aux | grep python)
- [ ] GPU wird genutzt (nvidia-smi zeigt Speicher)
- [ ] Logs werden geschrieben (tail -f training.log)
- [ ] Keine Fehler in Logs
- [ ] Speicherplatz ausreichend (df -h)
- [ ] Checkpoints werden gespeichert
- [ ] Loss sinkt über Zeit
- [ ] Training-Geschwindigkeit stabil

---

## Quick Commands

```bash
# Alles auf einen Blick
ssh root@123.45.67.89 -p 22 << 'EOF'
echo "=== Training Status ==="
ps aux | grep train_miniseries | grep -v grep && echo "✓ Training läuft" || echo "✗ Training nicht aktiv"

echo -e "\n=== GPU Status ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

echo -e "\n=== Speicher ==="
df -h /root/ | tail -1

echo -e "\n=== Letzte Logs ==="
tail -5 /root/outputs/logs/training.log
EOF
```

---

**Status**: ✅ Monitoring bereit  
**Letzte Aktualisierung**: 2026-01-13
