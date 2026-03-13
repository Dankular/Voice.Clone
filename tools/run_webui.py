import os
from argparse import ArgumentParser
from pathlib import Path

import gradio as gr
import pyrootutils
import torch
import uvicorn
from fastapi import FastAPI
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from tools.fish_api import init_api, router as api_router
from tools.webui import build_app
from tools.webui.inference import get_inference_wrapper

os.environ["EINX_FILTER_TRACEBACK"] = "false"

# Performance: enable TF32 and FP16 tensor cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--llama-checkpoint-path", type=Path, default="checkpoints/s2-pro")
    parser.add_argument("--decoder-checkpoint-path", type=Path, default="checkpoints/s2-pro/codec.pth")
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="light")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    if torch.backends.mps.is_available():
        args.device = "mps"
    elif torch.xpu.is_available():
        args.device = "xpu"
    elif not torch.cuda.is_available():
        args.device = "cpu"

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )

    logger.info("Loading VQ-GAN model...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    logger.info("Warming up...")
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )

    list(inference_engine.inference(ServeTTSRequest(
        text="Hello world.", references=[], reference_id=None,
        max_new_tokens=1024, chunk_length=200, top_p=0.7,
        repetition_penalty=1.5, temperature=0.7, format="wav",
    )))

    logger.info("Warming up done.")

    # Share engine with API
    init_api(inference_engine)

    inference_fct = get_inference_wrapper(inference_engine)

    # Build Gradio app
    gradio_blocks = build_app(inference_fct, args.theme)

    # Mount everything on FastAPI
    fastapi_app = FastAPI(title="Fish Speech", version="2.0")
    fastapi_app.include_router(api_router)

    app = gr.mount_gradio_app(fastapi_app, gradio_blocks, path="/")

    logger.info(f"Launching on http://{args.host}:{args.port}")
    logger.info("API docs: http://localhost:%d/docs", args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
