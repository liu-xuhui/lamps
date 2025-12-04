import matplotlib.pyplot as plt

def plot_weight_tiuta_sort(weight_tiuta_sort, threshold_index, delta_bar, epoch, first_k=20):

    plt.figure()
    plt.plot(weight_tiuta_sort[:first_k], marker='o', markerfacecolor='none', linestyle='none')
    plt.plot(threshold_index, weight_tiuta_sort[threshold_index], marker='o', markerfacecolor='red', linestyle='none')
    plt.axhline(y=delta_bar, color='r', linestyle='--', label='$\tau$')
    plt.ylabel("$\tilde{\Delta}$")
    plt.title(f"Epoch {epoch}")
    plt.show()