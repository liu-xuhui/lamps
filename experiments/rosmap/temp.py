import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
src_python = project_root / "src" / "python"
sys.path.insert(0, str(src_python))


# print(pd.read_csv("data/rosmap/rosmap_200.csv").iloc[:,1:].iloc[:, [12, 107, 129, 153]].columns)
# print(pd.read_csv("results/rosmap/rosmap_adamp_nonoracle_mr.csv"))



mr = pd.read_csv("results/rosmap/rosmap_adamp_nonoracle_mr.csv")

if mr.columns[0].startswith("Unnamed"):
    mr = mr.iloc[:, 1:]

counts = (
    mr.apply(pd.to_numeric, errors="coerce")   
      .stack()                                 
      .astype(int)                             
      .value_counts()                          
      .sort_values(ascending=False)            
)


feature_names = pd.read_csv("data/rosmap/rosmap_200.csv").iloc[:, 1:].columns


freq_table = pd.DataFrame({
    "feature_idx": counts.index,
    "feature_name": [feature_names[i] for i in counts.index],
    "frequency": counts.values
}).reset_index(drop=True)

print(freq_table)

