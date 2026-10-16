"""Make ``gqa`` importable however pytest is invoked (including --import-mode=importlib)."""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
