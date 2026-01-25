import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
print(project_root)
src_python = project_root / "src" / "python"
sys.path.insert(0, str(src_python))

from adamp.validation_methods import *

num_simus = 10
num_signals = 10
snr_list = [5, 10, 20]
corr_list = [0, 0.5, 0.9]
permute = 1
oracle = 1

total = len(snr_list) * num_simus * len(corr_list)

with tqdm(total=total, desc="Running simulations") as pbar:
    for snr in snr_list:
        for rep in range(num_simus):
            for corr in corr_list:
                df = pd.read_csv(f"data/simulation/linear_corr{corr}_snr{snr}_permute{permute}_rep{rep}.csv")

                X = df.iloc[:,:500].to_numpy()
                Y = df["Y"].values

                np.random.seed(110)

                selected, _ = LassoStabilitySelection(X, Y)
                out_df = pd.DataFrame()
                out_df["var"] = selected
                out_df.to_csv(f"temp_results/linear/stability_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv", index=False)

                selected, _ = compute_eBIC_select_features(X, Y)
                out_df = pd.DataFrame()
                out_df["var"] = selected
                out_df.to_csv(f"temp_results/linear/ebic_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv", index=False)

                selected, _ = compute_lassocv_select_features(X, Y)
                out_df = pd.DataFrame()
                out_df["var"] = selected
                out_df.to_csv(f"temp_results/linear/lassocv_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv", index=False)

                selected, _ = compute_lasso_oracle_select_features(X, Y)
                out_df = pd.DataFrame()
                out_df["var"] = selected
                out_df.to_csv(f"temp_results/linear/lassoor_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv", index=False)

                selected, _ = CPSS(X, Y)
                out_df = pd.DataFrame()
                out_df["var"] = selected
                out_df.to_csv(f"temp_results/linear/cpss_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv", index=False)

                selected, _ = Elastic(X, Y)
                out_df = pd.DataFrame()
                out_df["var"] = selected
                out_df.to_csv(f"temp_results/linear/elastic_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv", index=False)

                selected, _ = Elastic_oracle(X, Y)
                out_df = pd.DataFrame()
                out_df["var"] = selected
                out_df.to_csv(f"temp_results/linear/elasticor_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv", index=False)
                
                pbar.update(1)
            

