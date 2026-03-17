"""
Real-time inference script — low-latency, frame-by-frame generation.
Usage:
    python -m scripts.realtime_inference \
        --inference_config configs/inference/realtime.yaml \
        --result_dir results/realtime \
        [--version v15] \
        [--fps 25]
"""

import argparse
import os
import sys
import time

import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from musetalk_pipeline import (
    MuseTalkConfig,
    MuseTalkPipeline,
    extract_frames,
    load_audio_features,
    detect_face_bbox,
    make_mouth_mask,
)


def parse_args():
    p = argparse.ArgumentParser(description="MuseTalk real-time inference")
    p.add_argument("--inference_config", required=True)
    p.add_argument("--result_dir", default="results/realtime")
    p.add_argument("--unet_model_path", default=None)
    p.add_argument("--unet_config", default=None)
    p.add_argument("--version", choices=["v1", "v15"], default="v15")
    p.add_argument("--fps", type=int, default=25)
    return p.parse_args()


class RealtimePipeline(MuseTalkPipeline):
    """
    Extends the base pipeline with frame-by-frame streaming capability.
    After load_models(), call process_frame() for each incoming frame.
    """

    def process_frame(
        self,
        frame: np.ndarray,
        audio_embed: torch.Tensor,
        bbox_shift: int = 0,
    ) -> np.ndarray:
        """
        Process a single frame synchronously.

        Args:
            frame:       BGR uint8 (H, W, 3)
            audio_embed: (1, hidden_dim) Whisper feature for this frame
            bbox_shift:  vertical shift for face crop

        Returns:
            Lip-synced BGR frame.
        """
        bbox = detect_face_bbox(frame, bbox_shift)
        if bbox is None:
            return frame

        crop = self._preprocess_face_crop(frame, bbox).unsqueeze(0)  # (1,3,256,256)
        mask = make_mouth_mask(frame, bbox)

        latent = self._encode_latent(crop)
        mask_t = torch.from_numpy(
            cv2.resize(mask, (32, 32))
        ).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        masked_latent = latent * (1 - mask_t)

        audio_in = audio_embed.unsqueeze(0).to(self.device)  # (1, 1, hidden)
        pred_latent = self._run_unet(masked_latent, audio_in)
        pred_img = self._decode_latent(pred_latent)[0]

        face_bgr = self._postprocess_face_crop(pred_img)
        return self._paste_face(frame, face_bgr, bbox, mask[:, :, 0])


def main():
    args = parse_args()

    cfg = MuseTalkConfig(fps=args.fps)
    if args.version == "v15":
        cfg.unet_model_path = "models/musetalkV15/unet.pth"
        cfg.unet_config_path = "models/musetalkV15/musetalk.json"
    if args.unet_model_path:
        cfg.unet_model_path = args.unet_model_path
    if args.unet_config:
        cfg.unet_config_path = args.unet_config

    pipe = RealtimePipeline(cfg)
    pipe.load_models()

    tasks = OmegaConf.load(args.inference_config)
    os.makedirs(args.result_dir, exist_ok=True)

    for task_name, task in tasks.items():
        print(f"\nReal-time task: {task_name}")

        frames, orig_fps = extract_frames(task.video_path, args.fps)
        audio_features = load_audio_features(
            task.audio_path,
            pipe.whisper_processor,
            pipe.whisper_model,
            pipe.device,
        )
        n_frames = len(frames)
        audio_features = torch.nn.functional.interpolate(
            audio_features.T.unsqueeze(0),
            size=n_frames,
            mode="linear",
            align_corners=False,
        ).squeeze(0).T.unsqueeze(1)  # (n_frames, 1, hidden)

        video_stem = os.path.splitext(os.path.basename(task.video_path))[0]
        audio_stem = os.path.splitext(os.path.basename(task.audio_path))[0]
        tmp_path   = os.path.join(args.result_dir, f"{video_stem}_{audio_stem}_tmp.mp4")
        out_path   = os.path.join(args.result_dir, f"{video_stem}_{audio_stem}.mp4")

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            orig_fps,
            (w, h),
        )

        bbox_shift = task.get("bbox_shift", 0)
        t0 = time.perf_counter()
        for i, (frame, audio_embed) in enumerate(zip(frames, audio_features)):
            result = pipe.process_frame(frame, audio_embed, bbox_shift)
            writer.write(result)
            if i % 50 == 0:
                elapsed = time.perf_counter() - t0
                fps_achieved = (i + 1) / max(elapsed, 1e-6)
                print(f"  Frame {i}/{n_frames}  ({fps_achieved:.1f} fps)")

        writer.release()

        from musetalk_pipeline import merge_audio_video
        merge_audio_video(tmp_path, task.audio_path, out_path)
        os.unlink(tmp_path)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
