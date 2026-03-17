"""
Batch inference script — mirrors MuseTalk's scripts/inference.py
Usage:
    python -m scripts.inference \
        --inference_config configs/inference/test.yaml \
        --result_dir results/test \
        [--unet_model_path models/musetalk/pytorch_model.bin] \
        [--unet_config     models/musetalk/musetalk.json] \
        [--version         v1]
"""

import argparse
import sys
import os

# Ensure project root is on path when run as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from musetalk_pipeline import MuseTalkPipeline, MuseTalkConfig


def parse_args():
    p = argparse.ArgumentParser(description="MuseTalk batch inference")
    p.add_argument("--inference_config", required=True,
                   help="Path to YAML inference config")
    p.add_argument("--result_dir", default="results/test",
                   help="Directory for output videos")
    p.add_argument("--unet_model_path", default=None,
                   help="Override UNet weights path")
    p.add_argument("--unet_config", default=None,
                   help="Override UNet JSON config path")
    p.add_argument("--version", choices=["v1", "v15"], default="v1",
                   help="MuseTalk model version")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    # Build config
    cfg = MuseTalkConfig(version=args.version, batch_size=args.batch_size)

    if args.version == "v15":
        cfg.unet_model_path = "models/musetalkV15/unet.pth"
        cfg.unet_config_path = "models/musetalkV15/musetalk.json"
    if args.unet_model_path:
        cfg.unet_model_path = args.unet_model_path
    if args.unet_config:
        cfg.unet_config_path = args.unet_config

    # Load pipeline once, run all tasks
    pipe = MuseTalkPipeline(cfg)
    pipe.load_models()

    tasks = OmegaConf.load(args.inference_config)
    os.makedirs(args.result_dir, exist_ok=True)

    for task_name, task in tasks.items():
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"  video : {task.video_path}")
        print(f"  audio : {task.audio_path}")

        video_stem = os.path.splitext(os.path.basename(task.video_path))[0]
        audio_stem = os.path.splitext(os.path.basename(task.audio_path))[0]
        output_path = os.path.join(args.result_dir, f"{video_stem}_{audio_stem}.mp4")

        pipe.run(
            video_path=task.video_path,
            audio_path=task.audio_path,
            output_path=output_path,
            bbox_shift=task.get("bbox_shift", None),
        )

    print("\nAll tasks complete.")


if __name__ == "__main__":
    main()
