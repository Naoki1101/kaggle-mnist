from pathlib import Path

DATA_DIR = Path("./data")
INPUT_DATA_DIR = DATA_DIR / "input"
OUTPUT_DATA_DIR = DATA_DIR / "output"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TEMP_DATA_DIR = DATA_DIR / "temp"

CONFIG_DIR = Path("./configs")
LOG_DIR = Path("./logs")
NOTEBOOK_DIR = Path("./notebooks")

TARGET_COL = "label"
PIXEL_COLS = [f"pixel{i}" for i in range(28 ** 2)]
