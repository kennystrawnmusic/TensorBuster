import os
import re
import socket
import sys
import platform
import urllib.request
from html.parser import HTMLParser
from setuptools import build_meta as _orig
from setuptools.build_meta import *

# FORCE IPV4 ONLY (Fixes the getaddrinfo/IPv6 lab routing bug)
orig_getaddrinfo = socket.getaddrinfo
def forced_ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = forced_ipv4_getaddrinfo

# Try to import packaging tags
try:
    from packaging.tags import sys_tags
    SUPPORTED_TAGS = {f"{tag.interpreter}-{tag.abi}-{tag.platform}" for tag in sys_tags()}
except ImportError:
    SUPPORTED_TAGS = set()

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href':
                    self.links.append(value)

def _is_wheel_supported(filename):
    if not filename.endswith(".whl"):
        return False
        
    # 1. Always allow universal pure-python wheels
    if "none-any.whl" in filename:
        return True

    # 2. Autodetect the host OS/Platform string
    current_os = platform.system().lower() # 'windows', 'linux', 'darwin'
    
    # 3. Filter out completely mismatched OS wheels early
    if current_os == "windows" and "win_amd64" not in filename:
        return False
    elif current_os == "linux" and "linux_x86_64" not in filename:
        return False
    elif current_os == "darwin" and "macosx" not in filename:
        return False
        
    # 4. Fallback to PEP 425 tags verification if available
    if not SUPPORTED_TAGS:
        return True
        
    parts = filename[:-4].split('-')
    if len(parts) >= 3:
        wheel_tag = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
        return wheel_tag in SUPPORTED_TAGS
    return False

def _fetch_latest_wheel(package_name):
    # Detect if we need a specific CUDA build or a standard build based on OS
    # Note: PyTorch nightly uses 'cu130' for Windows/Linux, but macOS uses standard CPU/MPS paths
    current_os = platform.system().lower()
    
    if "torchao" in package_name:
        base_url = f"https://download-r2.pytorch.org/whl/nightly/{package_name}/"
    else:
        if current_os == "darwin":
            # macOS doesn't use the cu130 directory for nightlies
            base_url = f"https://download-r2.pytorch.org/whl/nightly/{package_name}/"
        else:
            base_url = f"https://download-r2.pytorch.org/whl/nightly/cu130/{package_name}/"
            
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36'}
    
    try:
        req = urllib.request.Request(base_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
            
        parser = LinkParser()
        parser.feed(html)
        
        valid_wheels = [link for link in parser.links if _is_wheel_supported(link)]
        
        if not valid_wheels:
            raise ValueError(f"No compatible {current_os} wheels found for {package_name}")
            
        latest_wheel = valid_wheels[-1]
        
        if latest_wheel.startswith('http://') or latest_wheel.startswith('https://'):
            return f"{package_name} @ {latest_wheel}"
        else:
            return f"{package_name} @ {base_url}{latest_wheel}"
        
    except Exception as e:
        raise RuntimeError(f"Failed to resolve compatible nightly wheel for {package_name}: {e}")

def _get_nightly_requirements():
    reqs = [
        _fetch_latest_wheel("torch"),
        _fetch_latest_wheel("torchvision"),
        _fetch_latest_wheel("torchaudio"),
        _fetch_latest_wheel("torchao")
    ]
    return reqs

def get_requires_for_build_wheel(config_settings=None):
    return _get_nightly_requirements() + _orig.get_requires_for_build_wheel(config_settings)

def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    # Determine the actual package name and version to construct the exact folder name
    # Since your package is 'tensorbuster' version '0.1.1+nightly20260619'
    # The directory name MUST be: tensorbuster-0.1.1+nightly20260619.dist-info
    
    # We try to use the original hook, but safely guard against NoneType
    try:
        dist_info_dir = _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
    except Exception:
        dist_info_dir = None

    if not dist_info_dir:
        # Fallback normalization rule for setuptools PEP 517 metadata directories:
        # name-version.dist-info (with dashes/spaces normalized to underscores)
        dist_info_dir = "tensorbuster-0.1.1+nightly20260619.dist-info"
        
    # Ensure the directory physically exists inside the target metadata temporary folder
    full_dist_info_path = os.path.join(metadata_directory, dist_info_dir)
    os.makedirs(full_dist_info_path, exist_ok=True)
    
    metadata_path = os.path.join(full_dist_info_path, "METADATA")
    
    # If setuptools didn't dump a base METADATA file, write a minimal header stub so we can append to it
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("Metadata-Version: 2.1\n")
            f.write("Name: tensorbuster\n")
            f.write("Version: 0.1.1+nightly20260619\n")

    # Read and patch the requirements
    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    nightly_names = ["torch", "torchvision", "torchaudio", "torchao"]
    new_lines = [l for l in lines if not any(f"Requires-Dist: {n}" in l for n in nightly_names)]
    
    for req in _get_nightly_requirements():
        new_lines.append(f"Requires-Dist: {req}\n")
        
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
        
    # Crucial: Must return a string representing the directory relative to metadata_directory
    return dist_info_dir
