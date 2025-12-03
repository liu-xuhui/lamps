import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

# go from data/simulation/generate_nonlinear_data.py to repo root
project_root = Path(__file__).resolve().parents[4]
src_python = project_root / "src" / "python"
sys.path.insert(0, str(src_python))

from adamp.validation_methods import Konckoff_select

num_simus = 30
snr_list = [4, 6, 8, 10]
corr_list = [0, 0.5, 0.9]
permute = 0

total = len(snr_list) * num_simus * len(corr_list)

with tqdm(total=total, desc="Running simulations") as pbar:
    for snr in snr_list:
        for rep in range(num_simus):
            for corr in corr_list:
                df = pd.read_csv(f"data/simulation/nonlinear_nonadditive_corr{corr}_snr{snr}_permute{permute}_rep{rep}.csv")

                X = df.iloc[:,:50].to_numpy()
                Y = df["Y"].values

                np.random.seed(110)

                selected = Konckoff_select(X, Y, 0.3)
                out_df = pd.DataFrame()
                out_df["var"] = selected
                out_df.to_csv(f"temp_results/nonlinear/nonadditive/knockoff03_selected_corr{corr}_snr{snr}_permute{permute}_rep{rep}.csv", index=False)

                selected = Konckoff_select(X, Y, 0.2)
                out_df = pd.DataFrame()
                out_df["var"] = selected
                out_df.to_csv(f"temp_results/nonlinear/nonadditive/knockoff02_selected_corr{corr}_snr{snr}_permute{permute}_rep{rep}.csv", index=False)

                selected = Konckoff_select(X, Y, 0.1)
                out_df = pd.DataFrame()
                out_df["var"] = selected
                out_df.to_csv(f"temp_results/nonlinear/nonadditive/knockoff01_selected_corr{corr}_snr{snr}_permute{permute}_rep{rep}.csv", index=False)

                pbar.update(1)
            

