from ACA_implementations import aca_tensor, aca_k_vectors, aca_matrix_x_vector, compare_cp_with_full
import plotting
import numpy as np

def amount_calculations(big_t, max_rank, random_seed, method=''):
    if method == "matrix" or method == "method1":
        aca_matrix_norms = aca_matrix_x_vector(big_t, max_rank, start_matrix=None, random_seed=random_seed)
    elif method == "k_hat" or method == "method3":
        return 0
    return 0


def main():
    path = "tensors/all_p_squat.npy"
    tensor = np.load(path)
    max_rank = 16
    random_seed = 0

    amount_calculations(tensor, max_rank, random_seed, method="matrix")


if __name__ == "__main__":
    main()