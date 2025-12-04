import os
import pickle
import pandas as pd

N = 200; M = 500; N1 = 50;

max_iter = 5

number_signals = 10
num_simus = 10
snr_list = [0.5, 1, 2, 5]
corr_list = [0, 0.5, 0.9]
result_dict = {0:{}, 0.5:{}, 0.9:{}}
folder = "temp_results/nonlinear/additive"

for permute in [0,1]:
    for oracle in [0,1]:
        for corr in corr_list:
            # com_method_result_dict = {"adammars":{}, "adamspam":{}, "spam":{}, "mars":{}, "kf1":{}, "kf2":{}, "kf3":{}, "hsic":{}}
            com_method_result_dict = {"adammars":{}, "adamspam":{}, "spam":{}, "mars":{}, "kf1":{}, "kf2":{}, "kf3":{}, "hsic":{}}
            # temporary, delete after running adam
            file_path = os.path.join("temp_results/temp_compare_result", f"adamars_nonlinear_add_oracle{oracle}_permute{permute}_result_dict.pkl")
            if os.path.exists(file_path):
                with open(f"temp_results/temp_compare_result/adamars_nonlinear_add_oracle{oracle}_permute{permute}_result_dict.pkl", "rb") as f:
                    temp_result_dict = pickle.load(f)
                    com_method_result_dict["adammars"] = temp_result_dict[corr]["adammars"]

            file_path = os.path.join("temp_results/temp_compare_result", f"adaspam_nonlinear_add_oracle{oracle}_permute{permute}_result_dict.pkl")
            if os.path.exists(file_path):
                with open(f"temp_results/temp_compare_result/adaspam_nonlinear_add_oracle{oracle}_permute{permute}_result_dict.pkl", "rb") as f:
                    temp_result_dict = pickle.load(f)
                    com_method_result_dict["adamspam"] = temp_result_dict[corr]["adaspam"]
            for snr in snr_list:
                com_method_result_dict["spam"][snr] = {}
                com_method_result_dict["mars"][snr] = {}
                com_method_result_dict["kf1"][snr] = {}
                com_method_result_dict["kf2"][snr] = {}
                com_method_result_dict["kf3"][snr] = {}
                com_method_result_dict["hsic"][snr] = {}
                for rep in range(num_simus):
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

        with open(f"results/nonlinear/additive/nonlinear_additive_oracle{oracle}_permute{permute}_result_dict.pkl", "wb") as f:
            pickle.dump(result_dict, f)
