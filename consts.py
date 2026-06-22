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

from middleware import SessionContextManager, SessionTracker, HFChatTemplatePreprocessor, ChatStateSaver, DynamicHostPortTracker, StegoWrapper

# Needs to be global to allow access from multiple Python modules
MCP_SERVER = FastMCP("TensorBuster C2 Server")

# Sliver-style session tracking
SESSIONS = []
SELECTED_SESSION = ""

# Model configuration
BASE_MODEL_ID = "NexVeridian/Qwen3-Coder-Next-8bit"
MODEL_PATH = Path("tb-base-model")