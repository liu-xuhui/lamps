import os
import pickle
import pandas as pd

N = 200; M = 500; N1 = 50;

max_iter = 5

number_signals = 10
num_simus = 10
snr_list = [0.5, 1, 2, 5]
corr_list = [0, 0.5, 0.9]
result_dict = {0.5:{}}
folder = "temp_results/nonlinear/additive"
hypertuning_folder = "temp_results/hyperparam_tuning/nonlinear_additive_corr0.5_permute0/best"

for permute in [0]:
    for oracle in [0]:
        for corr in corr_list:
            com_method_result_dict = {"adammars":{}, "adamspam":{}, "spam":{}, "mars":{}, "kf1":{}, "kf2":{}, "kf3":{}, "hsic":{}}
            for snr in snr_list:
                com_method_result_dict["adammars"][snr] = {}
                com_method_result_dict["adamspam"][snr] = {}
                com_method_result_dict["spam"][snr] = {}
                com_method_result_dict["mars"][snr] = {}
                com_method_result_dict["kf1"][snr] = {}
                com_method_result_dict["kf2"][snr] = {}
                com_method_result_dict["kf3"][snr] = {}
                com_method_result_dict["hsic"][snr] = {}
                for rep in range(num_simus):
                    file_path = os.path.join(hypertuning_folder, f"adammars_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["adammars"][snr][rep] = vars_list

                    file_path = os.path.join(hypertuning_folder, f"adamspam_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["adamspam"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"spam_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["spam"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"mars_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["mars"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"knockoff01_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["kf1"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"knockoff02_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["kf2"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"knockoff03_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["kf3"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"hsic_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["hsic"][snr][rep] = vars_list

            result_dict[corr] = com_method_result_dict

        with open(f"results/hyperparam_tuning/nonlinear_additive_oracle{oracle}_permute{permute}_result_dict.pkl", "wb") as f:
            pickle.dump(result_dict, f)
