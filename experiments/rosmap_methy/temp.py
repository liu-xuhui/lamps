import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
src_python = project_root / "src" / "python"
sys.path.insert(0, str(src_python))


print(pd.read_csv("data/rosmap_methy/rosmap_methy_200.csv").iloc[:,1:].iloc[:, [103, 171]].columns)

