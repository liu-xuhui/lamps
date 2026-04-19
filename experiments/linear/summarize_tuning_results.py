import os
import pickle
import pandas as pd

N = 200; M = 500; N1 = 50;

max_iter = 5

number_signals = 10
num_simus = 10
snr_list = [5, 10, 20, 30, 40, 50, 60]
corr_list = [0, 0.5, 0.9]
result_dict = {0.5:{}}
folder = "temp_results/linear"
permute_values = [int(os.environ.get("ADAMP_PERMUTE"))] if os.environ.get("ADAMP_PERMUTE") is not None else [0]

for permute in permute_values:
    hypertuning_folder = f"temp_results/hyperparam_tuning/linear_corr0.5_permute{permute}/best"
    for oracle in [1]:
        for corr in corr_list:
            com_method_result_dict = {"adamlinear":{}, "stability":{}, "ebic":{}, "lassocv":{}, "lassoor":{}, "cpss":{}, "elastic":{}, "elasticor":{}}
            
            for snr in snr_list:
                com_method_result_dict["adamlinear"][snr] = {}
                com_method_result_dict["stability"][snr] = {}
                com_method_result_dict["ebic"][snr] = {}
                com_method_result_dict["lassocv"][snr] = {}
                com_method_result_dict["lassoor"][snr] = {}
                com_method_result_dict["cpss"][snr] = {}
                com_method_result_dict["elastic"][snr] = {}
                com_method_result_dict["elasticor"][snr] = {}
                for rep in range(num_simus):
                    file_path = os.path.join(hypertuning_folder, f"adamlinear_selected_corr{corr}_snr{snr}_permute{permute}_oracle0_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["adamlinear"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"stability_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["stability"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"ebic_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["ebic"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"lassocv_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["lassocv"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"lassoor_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["lassoor"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"cpss_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["cpss"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"elastic_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["elastic"][snr][rep] = vars_list

                    file_path = os.path.join(folder, f"elasticor_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        vars_list = df["var"].dropna().astype(int).tolist() if "var" in df.columns else []
                    else:
                        vars_list = []
                    com_method_result_dict["elasticor"][snr][rep] = vars_list

            result_dict[corr] = com_method_result_dict

        with open(f"results/hyperparam_tuning/linear_oracle{oracle}_permute{permute}_result_dict.pkl", "wb") as f:
            pickle.dump(result_dict, f)
