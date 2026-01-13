# Data Upload Guide für RunPod

**Ziel**: 29 Datendateien auf RunPod A100 hochladen  
**Zielverzeichnis**: `/root/data/training_data/`  
**Geschätzte Upload-Zeit**: 10-30 Minuten (abhängig von Dateigröße und Internetgeschwindigkeit)

---

## Schritt 1: Pod-Verbindung vorbereiten

### 1.1: SSH-Verbindungsdaten sammeln

Auf RunPod Dashboard:
1. Gehen Sie zu **"Pods"** → **"Manage"**
2. Wählen Sie Ihre laufende Pod
3. Klicken Sie auf **"Connect"**
4. Sie sehen SSH-Befehl: `ssh root@<IP> -p <PORT>`

**Beispiel:**
```
ssh root@123.45.67.89 -p 22
```

Notieren Sie sich:
- **IP-Adresse**: `123.45.67.89`
- **Port**: `22`

### 1.2: SSH-Test

Öffnen Sie Terminal/PowerShell auf Ihrem Computer:

```bash
# Test SSH connection
ssh root@123.45.67.89 -p 22

# You should see a prompt like: root@pod-xyz:~#
# If yes, connection works!
# Type: exit
```

---

## Schritt 2: Datendateien vorbereiten

### 2.1: Lokale Daten organisieren

Auf Ihrem Computer:

```bash
# Erstellen Sie ein Verzeichnis mit allen 29 Dateien
mkdir -p ~/training_data_upload
cd ~/training_data_upload

# Kopieren Sie alle 29 Datendateien hierher
# Beispiel:
cp ~/Documents/data_file_1.txt .
cp ~/Documents/data_file_2.txt .
# ... (alle 29 Dateien)

# Verifizieren Sie die Anzahl
ls -1 | wc -l  # Sollte 29 sein
```

### 2.2: Dateigröße prüfen

```bash
# Gesamtgröße anschauen
du -sh ~/training_data_upload

# Einzelne Dateien
ls -lh ~/training_data_upload
```

---

## Schritt 3: Upload mit SCP (Empfohlen)

### 3.1: Einfacher Upload

```bash
# Auf Ihrem lokalen Computer:
scp -P 22 -r ~/training_data_upload/* root@123.45.67.89:/root/data/training_data/

# Beispiel mit echten Werten:
scp -P 22 -r ~/training_data_upload/* root@123.45.67.89:/root/data/training_data/
```

**Erklärung:**
- `-P 22` = Port (Standard ist 22)
- `-r` = Rekursiv (Verzeichnisse und Dateien)
- `~/training_data_upload/*` = Quelle (alle Dateien)
- `root@123.45.67.89:/root/data/training_data/` = Ziel

### 3.2: Upload mit Fortschrittsanzeige

Falls Sie `rsync` haben (besserer Fortschritt):

```bash
rsync -avz -e "ssh -p 22" ~/training_data_upload/ root@123.45.67.89:/root/data/training_data/

# Flags:
# -a = Archive mode (preserves permissions)
# -v = Verbose (shows progress)
# -z = Compress during transfer
```

### 3.3: Upload in mehreren Teilen (bei großen Dateien)

Falls einzelne Dateien sehr groß sind:

```bash
# Upload Datei für Datei
for file in ~/training_data_upload/*; do
    echo "Uploading $(basename $file)..."
    scp -P 22 "$file" root@123.45.67.89:/root/data/training_data/
done
```

---

## Schritt 4: Upload verifizieren

### 4.1: Auf RunPod prüfen

```bash
# SSH in die Pod
ssh root@123.45.67.89 -p 22

# Auf der Pod:
ls -lh /root/data/training_data/

# Anzahl der Dateien prüfen
ls -1 /root/data/training_data/ | wc -l  # Sollte 29 sein

# Gesamtgröße prüfen
du -sh /root/data/training_data/

# Zeilenanzahl prüfen (falls Text-Dateien)
wc -l /root/data/training_data/*

# Exit
exit
```

### 4.2: Checksums verifizieren (Optional)

```bash
# Auf Ihrem Computer:
md5sum ~/training_data_upload/* > checksums.txt

# Auf RunPod:
scp -P 22 checksums.txt root@123.45.67.89:/root/data/

# SSH in Pod:
ssh root@123.45.67.89 -p 22
cd /root/data/training_data
md5sum -c /root/data/checksums.txt

# Alle sollten "OK" sein
```

---

## Schritt 5: Datenformat validieren

### 5.1: Dateiformat prüfen

```bash
# SSH in Pod:
ssh root@123.45.67.89 -p 22

# Prüfen Sie das Format der Dateien:
file /root/data/training_data/*

# Erste Zeilen anschauen:
head -5 /root/data/training_data/data_file_1.txt

# Letzte Zeilen:
tail -5 /root/data/training_data/data_file_1.txt
```

### 5.2: Datenqualität prüfen

```bash
# Python-Script zum Validieren:
python3 << 'EOF'
import os
import glob

data_dir = "/root/data/training_data"
files = glob.glob(f"{data_dir}/*")

print(f"Total files: {len(files)}")
print(f"Total size: {sum(os.path.getsize(f) for f in files) / 1e9:.2f} GB")

for f in sorted(files)[:5]:
    size = os.path.getsize(f) / 1e6
    print(f"  {os.path.basename(f)}: {size:.2f} MB")

print("  ...")
EOF
```

---

## Troubleshooting

### Problem: "Permission denied"

**Lösung:**
```bash
# Auf RunPod, erstellen Sie das Verzeichnis:
mkdir -p /root/data/training_data
chmod 755 /root/data/training_data

# Dann Upload erneut versuchen
```

### Problem: "Connection timeout"

**Lösung:**
```bash
# SSH-Verbindung mit längerer Timeout:
scp -P 22 -o ConnectTimeout=30 -o StrictHostKeyChecking=no \
    -r ~/training_data_upload/* root@123.45.67.89:/root/data/training_data/
```

### Problem: "Slow upload"

**Lösung:**
```bash
# Komprimieren vor Upload:
tar -czf training_data.tar.gz ~/training_data_upload/
scp -P 22 training_data.tar.gz root@123.45.67.89:/root/data/

# Auf RunPod:
ssh root@123.45.67.89 -p 22
cd /root/data
tar -xzf training_data.tar.gz
mv training_data_upload/* training_data/
```

### Problem: "Disk space full"

**Lösung:**
```bash
# Auf RunPod, prüfen Sie Speicher:
df -h /root/data

# Falls voll, löschen Sie alte Dateien:
rm -rf /root/data/old_data/*
```

---

## Geschätzte Zeiten

| Dateigröße | Upload-Zeit | Verbindung |
|-----------|------------|-----------|
| 1 GB | 2-5 min | 100 Mbps |
| 5 GB | 10-25 min | 100 Mbps |
| 10 GB | 20-50 min | 100 Mbps |
| 50 GB | 100-250 min | 100 Mbps |

---

## Nach dem Upload

Sobald Daten hochgeladen sind:

1. ✅ Daten verifiziert
2. ✅ Format validiert
3. ✅ Training kann starten

**Starten Sie Training mit:**
```bash
ssh root@123.45.67.89 -p 22
cd ~/nanoGpt-Deepall-Agent
bash ~/start_training.sh
```

---

## Quick Reference

```bash
# Upload (schnell)
scp -P 22 -r ~/training_data_upload/* root@123.45.67.89:/root/data/training_data/

# Verifizieren
ssh root@123.45.67.89 -p 22 "ls -1 /root/data/training_data/ | wc -l"

# Training starten
ssh root@123.45.67.89 -p 22 "bash ~/start_training.sh"

# Logs anschauen
ssh root@123.45.67.89 -p 22 "tail -f /root/outputs/logs/training.log"
```

---

**Status**: ✅ Bereit zum Upload  
**Letzte Aktualisierung**: 2026-01-13
