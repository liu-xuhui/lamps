import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
src_python = project_root / "src" / "python"
sys.path.insert(0, str(src_python))

from adamp.validation_methods import HSIC_select

num_simus = 10
num_signals = 10
snr_list = [0.5, 1, 2, 5]
corr_list = [0, 0.5, 0.9]
permute = 0
oracle = 1 # oracle can only be 1

total = len(snr_list) * num_simus * len(corr_list)

with tqdm(total=total, desc="Running simulations") as pbar:
    for snr in snr_list:
        for rep in range(num_simus):
            for corr in corr_list:
                df = pd.read_csv(f"data/simulation/nonlinear_additive_corr{corr}_snr{snr}_permute{permute}_rep{rep}.csv")

                X = df.iloc[:,:500].to_numpy()
                Y = df["Y"].values

                np.random.seed(110)

                selected = HSIC_select(X, Y, num_signals)
                out_df = pd.DataFrame()
                out_df["var"] = selected
                out_df.to_csv(f"temp_results/nonlinear/additive/hsic_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv", index=False)

                
                pbar.update(1)
            

