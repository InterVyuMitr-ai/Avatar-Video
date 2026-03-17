"""
MuseTalk Pipeline
Audio-driven lip-sync for talking-head video generation.

Architecture:
  - Audio encoder:    OpenAI Whisper-tiny (frozen)
  - Image VAE:        stabilityai/sd-vae-ft-mse (frozen)
  - Generation UNet:  trained MuseTalk UNet (audio fused via cross-attention)
  - Face detection:   DWPose landmark detector
  - Face parsing:     BiSeNet (for mouth mask generation)
"""

from __future__ import annotations

import os
import cv2
import torch
import numpy as np
import soundfile as sf
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from diffusers import AutoencoderKL
from transformers import WhisperProcessor, WhisperModel
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class MuseTalkConfig:
    # Paths
    unet_model_path: str = "models/musetalk/pytorch_model.bin"
    unet_config_path: str = "models/musetalk/musetalk.json"
    vae_model_path: str = "models/sd-vae"
    whisper_model_path: str = "models/whisper"
    dwpose_model_path: str = "models/dwpose/dw-ll_ucoco_384.pth"
    face_parse_model_path: str = "models/face-parse-bisent/79999_iter.pth"

    # Inference
    version: str = "v1"          # "v1" or "v15"
    bbox_shift: int = 0          # vertical shift of face crop, tune per video
    fps: int = 25
    batch_size: int = 8

    # Device
    device: str = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, fps: int) -> tuple[list[np.ndarray], float]:
    """Return (list_of_BGR_frames, original_fps)."""
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, orig_fps


def extract_audio(video_path: str, out_wav: str) -> None:
    """Extract audio track from video to WAV using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", out_wav],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def merge_audio_video(video_no_audio: str, audio_path: str, output_path: str) -> None:
    """Combine silent video with audio into final output."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_no_audio,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def load_audio_features(
    audio_path: str,
    processor: WhisperProcessor,
    model: WhisperModel,
    device: str,
) -> torch.Tensor:
    """
    Encode audio with Whisper and return per-frame feature tensor.
    Shape: (T, seq_len, hidden_dim)
    """
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    inputs = processor(
        wav,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(device)

    with torch.no_grad():
        encoder_outputs = model.encoder(inputs)
    # (1, seq_len, hidden_dim) → (seq_len, hidden_dim)
    return encoder_outputs.last_hidden_state.squeeze(0).cpu()


# ---------------------------------------------------------------------------
# Face detection & mouth masking (placeholder wrappers)
# ---------------------------------------------------------------------------
# In production these delegate to DWPose + BiSeNet from the musetalk package.
# Replace the bodies with actual mmpose / bisenet calls once deps are installed.

def detect_face_bbox(frame: np.ndarray, bbox_shift: int = 0) -> Optional[tuple[int, int, int, int]]:
    """
    Return (x1, y1, x2, y2) face bounding box, or None if no face detected.
    Placeholder: uses OpenCV's Haar cascade for demo purposes.
    Production: replace with DWPose landmark-based detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    # Apply vertical shift
    y = max(0, y + bbox_shift)
    return x, y, x + w, y + h


def make_mouth_mask(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    Return a binary mask (H, W, 1) with the mouth region set to 1.
    Placeholder: masks the lower-third of the face crop.
    Production: replace with BiSeNet face parsing (label 11/12/13 = lips/teeth).
    """
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    mask = np.zeros((h, w, 1), dtype=np.float32)
    # Approximate mouth region: lower 40% of the face crop
    mouth_top = int(h * 0.60)
    mask[mouth_top:, :] = 1.0
    return mask


# ---------------------------------------------------------------------------
# MuseTalk UNet loader
# ---------------------------------------------------------------------------

def load_unet(config_path: str, weights_path: str, device: str):
    """
    Load the MuseTalk UNet from its JSON config + weights file.
    Returns the model in eval mode.
    """
    import json
    from diffusers import UNet2DConditionModel

    with open(config_path) as f:
        unet_config = json.load(f)

    unet = UNet2DConditionModel(**unet_config)

    state_dict = torch.load(weights_path, map_location="cpu")
    # Some checkpoints are wrapped under a key
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    unet.load_state_dict(state_dict, strict=False)
    unet.to(device).eval()
    return unet


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class MuseTalkPipeline:
    """
    End-to-end MuseTalk lip-sync pipeline.

    Usage:
        pipe = MuseTalkPipeline(MuseTalkConfig())
        pipe.load_models()
        pipe.run(
            video_path="data/video/source.mp4",
            audio_path="data/audio/target.wav",
            output_path="results/output.mp4",
        )
    """

    def __init__(self, config: MuseTalkConfig):
        self.cfg = config
        self.device = config.device
        self.unet = None
        self.vae = None
        self.whisper_processor = None
        self.whisper_model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        print(f"[MuseTalk] Loading models on device: {self.device}")

        # VAE
        print("  Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(self.cfg.vae_model_path)
        self.vae.to(self.device).eval()

        # Whisper
        print("  Loading Whisper-tiny...")
        self.whisper_processor = WhisperProcessor.from_pretrained(self.cfg.whisper_model_path)
        self.whisper_model = WhisperModel.from_pretrained(self.cfg.whisper_model_path)
        self.whisper_model.to(self.device).eval()

        # UNet
        print("  Loading MuseTalk UNet...")
        self.unet = load_unet(
            self.cfg.unet_config_path,
            self.cfg.unet_model_path,
            self.device,
        )

        print("[MuseTalk] All models loaded.")

    # ------------------------------------------------------------------
    # Frame preprocessing / postprocessing
    # ------------------------------------------------------------------

    def _preprocess_face_crop(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> torch.Tensor:
        """Crop face, resize to 256x256, normalize to [-1, 1]."""
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (256, 256))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(crop).float().permute(2, 0, 1) / 127.5 - 1.0
        return tensor  # (3, 256, 256)

    def _postprocess_face_crop(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert [-1,1] tensor back to BGR uint8 image at 256x256."""
        img = (tensor.clamp(-1, 1) + 1.0) * 127.5
        img = img.permute(1, 2, 0).byte().cpu().numpy()
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _paste_face(
        self,
        frame: np.ndarray,
        generated_face: np.ndarray,
        bbox: tuple[int, int, int, int],
        mask: np.ndarray,
    ) -> np.ndarray:
        """Blend generated face back into the original frame using the mouth mask."""
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        gen_resized = cv2.resize(generated_face, (w, h))
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_3c = np.stack([mask_resized] * 3, axis=-1)

        out = frame.copy()
        out[y1:y2, x1:x2] = (
            gen_resized * mask_3c + frame[y1:y2, x1:x2] * (1 - mask_3c)
        ).astype(np.uint8)
        return out

    # ------------------------------------------------------------------
    # Latent encoding / decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_latent(self, tensor: torch.Tensor) -> torch.Tensor:
        """Encode image tensor (B, 3, H, W) in [-1,1] → latent (B, 4, H/8, W/8)."""
        return self.vae.encode(tensor.to(self.device)).latent_dist.sample() * 0.18215

    @torch.no_grad()
    def _decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent (B, 4, H/8, W/8) → image (B, 3, H, W) in [-1,1]."""
        return self.vae.decode(latent / 0.18215).sample

    # ------------------------------------------------------------------
    # UNet inference (single batch)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_unet(
        self,
        masked_latents: torch.Tensor,   # (B, 4, 32, 32)
        audio_embeds: torch.Tensor,     # (B, seq, hidden)
    ) -> torch.Tensor:
        """
        Forward pass through the MuseTalk UNet.
        Returns predicted latents (B, 4, 32, 32).
        """
        timesteps = torch.zeros(masked_latents.shape[0], dtype=torch.long, device=self.device)
        # Concatenate masked latents with a blank noise channel as expected by the UNet
        # MuseTalk uses in_channels=8: 4 for masked image + 4 for mask
        noise = torch.zeros_like(masked_latents)
        unet_input = torch.cat([masked_latents, noise], dim=1)

        output = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states=audio_embeds.to(self.device),
        ).sample
        return output

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        bbox_shift: Optional[int] = None,
    ) -> str:
        """
        Run the full lip-sync pipeline.

        Args:
            video_path:  Path to source talking-head video.
            audio_path:  Path to target WAV audio.
            output_path: Where to write the output MP4.
            bbox_shift:  Override config bbox_shift for this run.

        Returns:
            Absolute path to the output file.
        """
        if self.unet is None:
            raise RuntimeError("Call load_models() before run().")

        shift = bbox_shift if bbox_shift is not None else self.cfg.bbox_shift
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 1. Extract source frames
        print("[MuseTalk] Extracting frames...")
        frames, orig_fps = extract_frames(video_path, self.cfg.fps)
        n_frames = len(frames)
        print(f"  {n_frames} frames @ {orig_fps:.1f} fps")

        # 2. Encode audio
        print("[MuseTalk] Encoding audio with Whisper...")
        audio_features = load_audio_features(
            audio_path, self.whisper_processor, self.whisper_model, self.device
        )
        # Interpolate audio features to match frame count
        audio_features = torch.nn.functional.interpolate(
            audio_features.T.unsqueeze(0),  # (1, hidden, seq)
            size=n_frames,
            mode="linear",
            align_corners=False,
        ).squeeze(0).T  # (n_frames, hidden)
        # Add sequence dim for cross-attention: (n_frames, 1, hidden)
        audio_features = audio_features.unsqueeze(1)

        # 3. Detect face bbox for each frame (reuse first detection for efficiency)
        print("[MuseTalk] Detecting face bounding boxes...")
        bboxes = []
        for frame in frames:
            bbox = detect_face_bbox(frame, shift)
            bboxes.append(bbox)

        # 4. Generate lip-synced frames in batches
        print("[MuseTalk] Running UNet inference...")
        result_frames = []
        batch = self.cfg.batch_size

        for i in range(0, n_frames, batch):
            batch_frames = frames[i : i + batch]
            batch_bboxes = bboxes[i : i + batch]
            batch_audio  = audio_features[i : i + batch]

            crops, masks, valid_idx = [], [], []
            for j, (frame, bbox) in enumerate(zip(batch_frames, batch_bboxes)):
                if bbox is None:
                    result_frames.append(frame)
                    continue
                crop_tensor = self._preprocess_face_crop(frame, bbox)
                mask        = make_mouth_mask(frame, bbox)
                crops.append(crop_tensor)
                masks.append(mask)
                valid_idx.append(j)

            if not crops:
                continue

            crop_batch  = torch.stack(crops)                    # (B, 3, 256, 256)
            audio_batch = batch_audio[valid_idx]                # (B, 1, hidden)

            # Encode to latent space
            latents     = self._encode_latent(crop_batch)       # (B, 4, 32, 32)

            # Zero out mouth region in latent (inpainting signal)
            mask_tensor = torch.from_numpy(
                np.stack([cv2.resize(m, (32, 32)) for m in masks])
            ).permute(0, 3, 1, 2).float().to(self.device)      # (B, 1, 32, 32)
            masked_latents = latents * (1 - mask_tensor)

            # UNet forward
            pred_latents = self._run_unet(masked_latents, audio_batch)

            # Decode predictions
            pred_images  = self._decode_latent(pred_latents)   # (B, 3, 256, 256)

            # Paste back into original frames
            for k, (j, pred_img) in enumerate(zip(valid_idx, pred_images)):
                frame     = batch_frames[j]
                bbox      = batch_bboxes[j]
                face_bgr  = self._postprocess_face_crop(pred_img)
                mask_2d   = masks[k][:, :, 0]
                blended   = self._paste_face(frame, face_bgr, bbox, mask_2d)
                result_frames.append(blended)

            if (i // batch) % 5 == 0:
                pct = min(100, int(i / n_frames * 100))
                print(f"  {pct}% ({i}/{n_frames} frames)")

        # 5. Write output video
        print("[MuseTalk] Writing output video...")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        h, w = result_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            orig_fps,
            (w, h),
        )
        for frame in result_frames:
            writer.write(frame)
        writer.release()

        # 6. Mux audio
        merge_audio_video(tmp_path, audio_path, output_path)
        os.unlink(tmp_path)

        print(f"[MuseTalk] Done → {output_path}")
        return os.path.abspath(output_path)
