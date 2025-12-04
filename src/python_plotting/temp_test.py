# import pickle
# with open(f"results/nonlinear/additive/nonlinear_additive_oracle{1}_permute{1}_result_dict.pkl", "rb") as f:
#     result_dict = pickle.load(f)

# print(result_dict[0.5]["hsic"][5][5])


import os
import re

folder = "temp_results/nonlinear/nonadditive"

# allow corr and snr to be integers or decimals, e.g. corr0, corr0.5, snr1, snr0.9
pattern = re.compile(
    r"(hsic_selected_corr\d+\.?\d*_snr\d+\.?\d*_permute\d+)"
    r"(_rep\d+)(\.csv)$"
)

for fname in os.listdir(folder):
    m = pattern.match(fname)
    if m:
        new_name = f"{m.group(1)}_oracle1{m.group(2)}{m.group(3)}"
        old_path = os.path.join(folder, fname)
        new_path = os.path.join(folder, new_name)
        print(f"{fname}  -->  {new_name}")
        os.rename(old_path, new_path)