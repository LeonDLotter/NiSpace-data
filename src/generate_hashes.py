import os
import hashlib
import json
from pathlib import Path

# Directories to include in hash generation
DIRECTORIES = ['example', 'parcellation', 'reference', 'template']

# Output file for hashes
OUTPUT_FILE = 'file_hashes.json'

# Function to calculate the SHA-256 hash of a file
def calculate_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Function to check if a file should be ignored
def is_ignored(file_path):
    # Exclude .DS_Store files
    return file_path.endswith('.DS_Store')

# Updated function to generate hashes for all files in the specified directories
def generate_hashes():
    file_hashes = {}
    for directory in DIRECTORIES:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                # Skip .DS_Store files
                if is_ignored(file_path):
                    continue
                file_hashes[file_path] = calculate_file_hash(file_path)
    # Save the hashes to a JSON file
    with open(OUTPUT_FILE, 'w') as json_file:
        json.dump(file_hashes, json_file, indent=4)

if __name__ == "__main__":
    generate_hashes() 