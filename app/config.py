import os
from pathlib import Path

UPLOAD_DIR = Path("input/temp_upload")
OUTPUT_DIR = Path("output/result")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load các biến môi trường từ file .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("API_KEY")
