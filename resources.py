import torch
import os
import struct
import subprocess
import threading
import tempfile
import random
import json
import importlib
import copy
import io
import zipfile
import inspect

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
import numpy as np

from socket import gethostname, gethostbyname
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, AutoModel, AutoTokenizer
from fastmcp import FastMCP, Client, Context as ClientContext
from fastmcp.dependencies import CurrentContext, CurrentFastMCP
from fastmcp.server.context import Context as ServerContext
from fastmcp.server.dependencies import get_server, get_http_request
from fastmcp.server.middleware import Middleware
from argparse import ArgumentParser
from huggingface_hub import snapshot_download
from pathlib import Path

from consts import *

@MCP_SERVER.resource(f'weights://{BASE_MODEL_ID}')
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

@MCP_SERVER.resource(f'tokenizer://{BASE_MODEL_ID}')
def load_tokenizer(server: FastMCP = CurrentFastMCP()) -> AutoTokenizer:
    """
    Hosts the tokenizer for the currently running base model
    """

    # This is a piece of custom middleware I wrote myself, so should be loaded automatically into every instance
    session_context_middleware = next(m for m in server.middleware if "SessionContextManager" in m.name)

    return session_context_middleware.get_tokenizer()