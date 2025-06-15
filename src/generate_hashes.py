import os
import hashlib
import shutil

# Directories to include in hash generation (dynamic)
DIRECTORIES = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')]

# Hidden folder for shadow files
HASH_DIR = '.hash'

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
    isignored = False
    if any([file_path.endswith(ext) for ext in ['.DS_Store', '.hash']]):
        isignored = True
    elif any([file_path.startswith(prefix) for prefix in ['.', '_']]):
        isignored = True
    return isignored

# Function to create shadow files with hashes and remove shadows for deleted files
def create_shadow_files():
    # Create a set of all current file paths
    current_files = set()
    for directory in DIRECTORIES:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if not is_ignored(file_path):
                    current_files.add(file_path)

    # Create a set of all current shadow file paths
    current_shadows = set()
    if os.path.exists(HASH_DIR):
        for root, _, files in os.walk(HASH_DIR):
            for file in files:
                shadow_path = os.path.join(root, file)
                current_shadows.add(os.path.relpath(shadow_path, HASH_DIR))

    # Remove shadows for deleted files
    for shadow in current_shadows:
        if shadow not in current_files:
            os.remove(os.path.join(HASH_DIR, shadow))

    # Create or update shadows for current files
    for file_path in current_files:
        file_hash = calculate_file_hash(file_path)
        # Append '.hash' to the file path for the shadow file
        shadow_path = os.path.join(HASH_DIR, file_path + '.sha256')
        os.makedirs(os.path.dirname(shadow_path), exist_ok=True)
        with open(shadow_path, 'w') as shadow_file:
            shadow_file.write(file_hash)

if __name__ == "__main__":
    create_shadow_files() 