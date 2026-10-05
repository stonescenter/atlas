
import hashlib
from pathlib import Path

def generate_md5_id(input_string):
    # Encode the string to bytes, then compute the md5 hash
    encoded_string = input_string.encode('utf-8')
    md5_hash = hashlib.md5(encoded_string)
    
    # Return the 32-character hex string
    return md5_hash.hexdigest()


def ensure_directory(path: str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path