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

# Use uv if available (local), otherwise plain pip (Colab)
if command -v uv &>/dev/null; then
  uv pip install -q -U "huggingface_hub[cli]" gdown
else
  pip install -q -U "huggingface_hub[cli]" gdown
fi

# Use Python API directly — avoids PATH issues with huggingface-cli in Colab
hf_download() {
  # Usage: hf_download <repo_id> <local_dir> <pattern1> [pattern2 ...]
  REPO="$1"; LOCAL_DIR="$2"; shift 2
  PATTERNS=""
  for p in "$@"; do
    PATTERNS="${PATTERNS}'${p}',"
  done
  python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${REPO}",
    local_dir="${LOCAL_DIR}",
    allow_patterns=[${PATTERNS}],
)
PYEOF
}

# ---- MuseTalk v1.0 ----
echo "Downloading MuseTalk v1.0..."
hf_download TMElyralab/MuseTalk "$MODELS_DIR" \
  "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

# ---- MuseTalk v1.5 ----
echo "Downloading MuseTalk v1.5..."
hf_download TMElyralab/MuseTalk "$MODELS_DIR" \
  "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# ---- Stable Diffusion VAE ----
echo "Downloading SD VAE (ft-mse)..."
hf_download stabilityai/sd-vae-ft-mse "$MODELS_DIR/sd-vae" \
  "config.json" "diffusion_pytorch_model.bin"

# ---- Whisper-tiny ----
echo "Downloading Whisper-tiny..."
hf_download openai/whisper-tiny "$MODELS_DIR/whisper" \
  "config.json" "pytorch_model.bin" "preprocessor_config.json"

# ---- DWPose ----
echo "Downloading DWPose..."
hf_download yzd-v/DWPose "$MODELS_DIR/dwpose" \
  "dw-ll_ucoco_384.pth"

# ---- SyncNet (v1.5 sync loss) ----
echo "Downloading SyncNet..."
hf_download ByteDance/LatentSync "$MODELS_DIR/syncnet" \
  "latentsync_syncnet.pt"

# ---- Face Parsing BiSeNet ----
echo "Downloading face-parse-bisent..."
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
  -O "$MODELS_DIR/face-parse-bisent/79999_iter.pth"
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth"

echo ""
echo "All models downloaded to ./$MODELS_DIR/"
