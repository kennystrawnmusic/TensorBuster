from pathlib import Path

from fastmcp import FastMCP

# Needs to be global to allow access from multiple Python modules
MCP_SERVER = FastMCP("TensorBuster C2 Server")

# Sliver-style session tracking
SESSIONS = [""]
SELECTED_SESSION = ""

# Model configuration
BASE_MODEL_ID: str = "NexVeridian/Qwen3-Coder-Next-8bit"
MODEL_PATH = Path("tb-base-model")
