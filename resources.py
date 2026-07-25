import os
from pathlib import Path
from typing import cast

from fastmcp import FastMCP
from fastmcp.dependencies import CurrentFastMCP
from transformers import AutoModel, AutoTokenizer

from consts import *
from middleware import SessionContextManager
from tools import download_base_model


@MCP_SERVER.resource(f"weights://{BASE_MODEL_ID}")
def download_weights() -> AutoModel:
    """
    Hosts a binary copy of the base model weights on the MCP server as a resource for agents to download
    """

    model_path = Path("tb-base-model")

    if not os.path.exists("tb-base-model"):
        _, model = download_base_model({BASE_MODEL_ID}, str(model_path))
    else:
        print("Base model already exists, skipping server-side download")
        model = AutoModel.from_pretrained(model_path, local_files_only=True)

    return model


@MCP_SERVER.resource(f"tokenizer://{BASE_MODEL_ID}")
def load_tokenizer(server: FastMCP = CurrentFastMCP()) -> AutoTokenizer:
    """
    Hosts the tokenizer for the currently running base model
    """

    # This is a piece of custom middleware I wrote myself, so should be loaded automatically into every instance
    session_context_middleware = cast(SessionContextManager, next(
        m for m in server.middleware if "SessionContextManager" in m.__class__.__name__
    ))

    return session_context_middleware.get_tokenizer()
