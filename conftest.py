import sys
from pathlib import Path

# Ensure root directory is always at the top of sys.path for test discovery and imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
