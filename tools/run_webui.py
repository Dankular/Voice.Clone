import os
from argparse import ArgumentParser

import gradio as gr
import pyrootutils
import uvicorn
from fastapi import FastAPI
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.fish_api import router as api_router
from tools.webui import build_app
from tools.webui.inference import inference_wrapper


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--theme", type=str, default="light")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Build Gradio app
    gradio_blocks = build_app(inference_wrapper, args.theme)

    # Mount everything on FastAPI
    fastapi_app = FastAPI(title="Voice Clone — OmniVoice", version="2.0")
    fastapi_app.include_router(api_router)

    app = gr.mount_gradio_app(fastapi_app, gradio_blocks, path="/")

    logger.info(f"Launching on http://{args.host}:{args.port}")
    logger.info("API docs: http://localhost:%d/docs", args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
