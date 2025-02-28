#!/usr/bin/env python3
"""
Script to download the deepseek-r1 model for WebGPU.
This script will download the model files from Hugging Face and save them to the static/models directory.
"""

import os
import requests
import json
from tqdm import tqdm
import argparse

# Define the model repository and files to download
MODEL_REPO = "deepseek-ai/deepseek-coder-1.3b-instruct"
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors"
]

def download_file(url, dest_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(dest_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)

def main():
    parser = argparse.ArgumentParser(description="Download deepseek-r1 model for WebGPU")
    parser.add_argument("--output-dir", default="static/models/deepseek-r1", help="Output directory for model files")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Downloading model files from {MODEL_REPO} to {args.output_dir}...")
    
    # Download model files
    for file in MODEL_FILES:
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{file}"
        dest_path = os.path.join(args.output_dir, file)
        
        if os.path.exists(dest_path):
            print(f"File {file} already exists, skipping...")
            continue
        
        print(f"Downloading {file}...")
        download_file(url, dest_path)
    
    # Create a tokenizer_config.json file if it doesn't exist
    tokenizer_config_path = os.path.join(args.output_dir, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        print("Creating tokenizer_config.json...")
        with open(tokenizer_config_path, 'w') as f:
            json.dump({
                "model_type": "llama",
                "add_bos_token": True,
                "add_eos_token": False
            }, f, indent=2)
    
    print("Download complete!")

if __name__ == "__main__":
    main()
