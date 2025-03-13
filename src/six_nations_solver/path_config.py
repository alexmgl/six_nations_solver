from pathlib import Path

# Define the base directory as the parent of 'src'
BASE_DIR = Path(__file__).resolve().parents[2]

# Define the data directory path
DATA_DIR = BASE_DIR / "data"

# Ensure the data directory exists
DATA_DIR.mkdir(exist_ok=True)


