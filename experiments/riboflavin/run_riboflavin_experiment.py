import sys
from pathlib import Path

# Add src/python to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from adamp.adamp_main import *
from adamp.simulation_functions import *
