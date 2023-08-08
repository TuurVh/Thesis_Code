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

    plt.xlabel("# vectoren")
    plt.ylabel("Relatieve fout")

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=num_cols)
    plt.show()


def plot_aris(spec_data, medoid_data):

    means_spec = np.mean(spec_data, axis=1)
    std_devs_spec = np.std(spec_data, axis=1)

    means_medoid = np.mean(medoid_data, axis=1)
    std_devs_medoid = np.std(medoid_data, axis=1)

    # Set the number of parameter values and runs
    num_params = len(spec_data)
    num_runs = len(spec_data[0])

    # Set the width of the bars and the positions of the groups
    bar_width = 0.35
    index = np.arange(num_params)

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot the mean values as bars
    ax.bar(index - bar_width / 2, means_spec, bar_width, label='Spectral Clustering', yerr=std_devs_spec)
    ax.bar(index + bar_width / 2, means_medoid, bar_width, label='K-Medoids', yerr=std_devs_medoid)

    # Set the labels and title
    ax.set_xlabel('Rank')
    ax.set_ylabel('ARI-score')
    ax.set_title('Mean and Deviation of ARI-score for Different Ranks')
    ax.set_xticks(index)
    ax.set_xticklabels([f'{(i+1) * 5}' for i in range(num_params)])

    # Show the legend
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_rel_errs(data):

    means = np.mean(data, axis=1)
    std_devs = np.std(data, axis=1)
    # Set the number of parameter values and runs
    num_params = len(data)
    num_runs = len(data[0])

    # Set the width of the bars and the positions of the groups
    bar_width = 0.35
    index = np.arange(num_params)

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot the mean values as bars
    ax.bar(index, means, bar_width, label='matrix method', yerr=std_devs)

    # Set the labels and title
    ax.set_xlabel('Rank')
    ax.set_ylabel('Rel. error')
    ax.set_title('Mean and Deviation of Relative Error for increasing ranks')
    ax.set_xticks(index)
    ax.set_xticklabels([f'{(i+1) * 5}' for i in range(num_params)])

    # Show the legend
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_err_to_calcs(relative_errors, calculated_values):
    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot the relative errors as a line plot
    ax.plot(calculated_values, relative_errors, marker='o', linestyle='-', color='b')

    # Set the labels and title
    ax.set_xlabel('Calculated Values')
    ax.set_ylabel('Relative Error')
    ax.set_title('Relative Error vs. Calculated Values')

    # Display the plot
    plt.show()


def plot_err_to_calcs_percentage(relative_errors, calculated_values, p):
    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot the relative errors as a line plot
    calculated_values = [x / p for x in calculated_values]
    ax.plot(calculated_values, relative_errors, marker='o', linestyle='-', color='b')

    # Set the labels and title
    ax.set_xlabel('Calculated Values')
    ax.set_ylabel('Relative Error')
    ax.set_title('Relative Error vs. Calculated Values')

    # Display the plot
    plt.show()

