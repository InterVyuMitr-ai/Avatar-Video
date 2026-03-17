#!/bin/bash
# Downloads all required MuseTalk model weights

set -e

MODELS_DIR="models"

mkdir -p \
  "$MODELS_DIR/musetalk" \
  "$MODELS_DIR/musetalkV15" \
  "$MODELS_DIR/sd-vae" \
  "$MODELS_DIR/whisper" \
  "$MODELS_DIR/dwpose" \
  "$MODELS_DIR/syncnet" \
  "$MODELS_DIR/face-parse-bisent"

uv pip install -U "huggingface_hub[cli]" gdown -q

# ---- MuseTalk v1.0 ----
echo "Downloading MuseTalk v1.0..."
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir "$MODELS_DIR" \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

# ---- MuseTalk v1.5 ----
echo "Downloading MuseTalk v1.5..."
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir "$MODELS_DIR" \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# ---- Stable Diffusion VAE ----
echo "Downloading SD VAE (ft-mse)..."
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir "$MODELS_DIR/sd-vae" \
  --include "config.json" "diffusion_pytorch_model.bin"

# ---- Whisper-tiny ----
echo "Downloading Whisper-tiny..."
huggingface-cli download openai/whisper-tiny \
  --local-dir "$MODELS_DIR/whisper" \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# ---- DWPose ----
echo "Downloading DWPose..."
huggingface-cli download yzd-v/DWPose \
  --local-dir "$MODELS_DIR/dwpose" \
  --include "dw-ll_ucoco_384.pth"

# ---- SyncNet (v1.5 sync loss) ----
echo "Downloading SyncNet..."
huggingface-cli download ByteDance/LatentSync \
  --local-dir "$MODELS_DIR/syncnet" \
  --include "latentsync_syncnet.pt"

# ---- Face Parsing BiSeNet ----
echo "Downloading face-parse-bisent..."
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
  -O "$MODELS_DIR/face-parse-bisent/79999_iter.pth"
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth"

echo ""
echo "All models downloaded to ./$MODELS_DIR/"
