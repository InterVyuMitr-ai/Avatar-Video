"""
MuseTalk Gradio Web UI
Run:  python app.py
"""

import os
import tempfile
import gradio as gr

from musetalk_pipeline import MuseTalkPipeline, MuseTalkConfig

# ---------------------------------------------------------------------------
# Global pipeline (loaded once on startup)
# ---------------------------------------------------------------------------
_pipe: MuseTalkPipeline | None = None


def get_pipeline(version: str) -> MuseTalkPipeline:
    global _pipe
    cfg = MuseTalkConfig(version=version)
    if version == "v15":
        cfg.unet_model_path = "models/musetalkV15/unet.pth"
        cfg.unet_config_path = "models/musetalkV15/musetalk.json"
    _pipe = MuseTalkPipeline(cfg)
    _pipe.load_models()
    return _pipe


# ---------------------------------------------------------------------------
# Inference function wired to Gradio
# ---------------------------------------------------------------------------

def run_inference(
    video_file: str,
    audio_file: str,
    bbox_shift: int,
    version: str,
) -> str:
    pipe = get_pipeline(version)

    out_dir = "results/gradio"
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=out_dir, delete=False) as f:
        output_path = f.name

    pipe.run(
        video_path=video_file,
        audio_path=audio_file,
        output_path=output_path,
        bbox_shift=bbox_shift,
    )
    return output_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="MuseTalk — InterVyu.ai") as demo:
    gr.Markdown("# MuseTalk — Audio-Driven Lip Sync\nUpload a talking-head video and a target audio file.")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Source Video")
            audio_input = gr.Audio(label="Target Audio", type="filepath")
            bbox_shift   = gr.Slider(-15, 15, value=0, step=1,
                                     label="BBox Shift (tune if lips look off)")
            version      = gr.Radio(["v1", "v15"], value="v1",
                                    label="Model Version")
            run_btn      = gr.Button("Generate", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Result")

    run_btn.click(
        fn=run_inference,
        inputs=[video_input, audio_input, bbox_shift, version],
        outputs=video_output,
    )

if __name__ == "__main__":
    demo.launch(share=False)
