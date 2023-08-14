import matplotlib.pyplot as plt
import numpy as np

def plot_norms_aca_cp(amount_vects, aca_m=None, aca_k=None, aca_v=None, cp_norms=None):
    amount_vects = range(1, amount_vects+1)
    num_cols = 0
    plt.yscale('log')
    if aca_m is not None:
        plt.plot(amount_vects, aca_m, label='A - ACA_matrix')
        num_cols += 1
    if aca_k is not None:
        plt.plot(amount_vects, aca_k, label='A - ACA_k_hat')
        num_cols += 1
    if aca_v is not None:
        plt.plot(amount_vects, aca_v, label='A - ACA_vectors')
        num_cols += 1
    if cp_norms is not None:
        plt.plot(amount_vects, cp_norms, label='A - CP')
        num_cols += 1

    plt.xlabel("# iteraties")
    plt.ylabel("Relatieve fout")

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=num_cols)
    plt.show()


def plot_aris(spec_data, medoid_data):

    means_spec = np.mean(spec_data, axis=1)
    std_devs_spec = np.std(spec_data, axis=1)

    means_medoid = np.mean(medoid_data, axis=1)
    std_devs_medoid = np.std(medoid_data, axis=1)

    num_params = len(spec_data)

    bar_width = 0.35
    index = np.arange(num_params)

    fig, ax = plt.subplots()

    ax.bar(index - bar_width / 2, means_spec, bar_width, label='Spectral Clustering', yerr=std_devs_spec)
    ax.bar(index + bar_width / 2, means_medoid, bar_width, label='K-Medoids', yerr=std_devs_medoid)

    ax.set_xlabel('Rank')
    ax.set_ylabel('ARI-score')
    ax.set_title('Mean and Deviation of ARI-score for Different Ranks')
    ax.set_xticks(index)
    ax.set_xticklabels([f'{(i+1) * 5}' for i in range(num_params)])

    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_kmeans_aris(k_data, cp_data):

    means_spec = np.mean(k_data, axis=1)
    std_devs_spec = np.std(k_data, axis=1)

    num_params = len(k_data)

    bar_width = 0.35
    index = np.arange(num_params)

    fig, ax = plt.subplots()

    ax.bar(index - bar_width / 2, means_spec, bar_width, label='ACA-T', yerr=std_devs_spec)
    ax.bar(index + bar_width / 2, cp_data, bar_width, label='CP', color='tab:red')

    ax.set_xlabel('Rang')
    ax.set_ylabel('ARI-score')
    ax.set_title('ARI-score voor verschillende rangen, rijen als feature vectoren.')
    ax.set_xticks(index)
    ax.set_xticklabels([f'{(i*10) + 5}' for i in range(num_params)])

    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_rel_errs(data):
    means = np.mean(data, axis=1)
    std_devs = np.std(data, axis=1)

    num_params = len(data)

    bar_width = 0.35
    index = np.arange(num_params)

    fig, ax = plt.subplots()

    ax.bar(index, means, bar_width, label='Matrix methode', yerr=std_devs, color='tab:orange')

    ax.set_xlabel('Rang')
    ax.set_ylabel('Relatieve fout')
    ax.set_title('Relatieve fout van de benadering per rang')
    ax.set_xticks(index)
    ax.set_xticklabels([f'{(i+1) * 5}' for i in range(num_params)])

    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_rel_errs_cp(data):

    num_params = len(data)

    bar_width = 0.35
    index = np.arange(num_params)

    fig, ax = plt.subplots()

    ax.bar(index, data, bar_width, label='CP decompositie', color='tab:red')

    ax.set_xlabel('Rang')
    ax.set_ylabel('Relatieve fout')
    ax.set_title('Relatieve fout van de benadering per rang')
    ax.set_xticks(index)
    ax.set_xticklabels([f'{(i+1) * 5}' for i in range(num_params)])

    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_rank_to_calcs(tensor, max_rank, step):
    z, y, x = tensor.shape

    mat_calcs = []
    vect_calcs = []
    for r in range(5, max_rank+1, step):
        amount_mat_calcs = (((x * (x + 1)) / 2) + z) * r
        amount_vect_calcs = ((x + y) + z) * r
        total = (((x*(x+1))/2) * z)
        mat_calcs.append(amount_mat_calcs/total)
        vect_calcs.append(amount_vect_calcs/total)

    num_params = len(vect_calcs)

    bar_width = 0.35
    index = np.arange(num_params)

    fig, ax = plt.subplots()

    ax.bar(index, vect_calcs, bar_width, label='Vectoren methode')
    # ax.bar(index, mat_calcs, bar_width, label='Matrix methode', color='tab:orange')

    ax.set_xlabel('Rang')
    ax.set_ylabel('Relatief aantal DTW berekeningen')
    ax.set_title('Relatief aantal DTW berekeningen nodig per rang')
    ax.set_xticks(index)
    ax.set_xticklabels([f'{(i+1) * 5}' for i in range(num_params)])

    ax.legend()
    plt.yscale('log')

    plt.tight_layout()
    plt.show()


def plot_err_to_calcs(relative_errors, calculated_values):
    fig, ax = plt.subplots()

    ax.plot(calculated_values, relative_errors, marker='o', linestyle='-', color='tab:orange')

    ax.set_xlabel('Aantal DTW berekeningen')
    ax.set_ylabel('Relatieve fout')
    ax.set_title('Relatieve fout bij een toenemend aantal DTW berekeningen')
    plt.yscale('log')

    plt.show()


def plot_err_to_calcs_percentage(relative_errors, calculated_values, p):
    fig, ax = plt.subplots()

    calculated_values = [x / p for x in calculated_values]
    ax.plot(calculated_values, relative_errors, marker='o', linestyle='-', color='tab:orange')
    # ax.plot(calculated_values, relative_errors, marker='o', linestyle='-', color='tab:orange')

    plt.yscale('log')

    ax.set_xlabel('Relatief aantal DTW berekeningen')
    ax.set_ylabel('Relatieve fout')
    ax.set_title('Relatieve fout bij een toenemend relatief aantal DTW berekeningen')

    plt.show()

