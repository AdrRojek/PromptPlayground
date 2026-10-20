"""Environment settings."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
