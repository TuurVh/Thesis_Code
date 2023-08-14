from ACA_T_matrix import aca_matrix_x_vector
from ACA_T_vectors import aca_tensor as aca_t
from tests import plot_utils
import numpy as np
from tests.test_utils import tensor_symmetric_test

# Set print to print matrices and vectors in one line
np.set_printoptions(linewidth=np.inf, suppress=True)

def random_tensor(shape, low, high, seed):
    np.random.seed(seed)

    tensor = np.zeros(shape)
    for i in range(shape[0]):
        # Generate a random symmetric 3-by-3 matrix
        sym_matrix = np.random.randint(low, high, size=(shape[1], shape[1]))
        sym_matrix = (sym_matrix + sym_matrix.T) // 2
        np.fill_diagonal(sym_matrix, 0)

        tensor[i, :, :] = sym_matrix

    return tensor


def compare_acas(big_t, max_rank, random_seed, plot=False):
    aca_vects_norms = None
    aca_k_hat_norms = None
    aca_matrix_norms = None
    cp_norms = None
    # ACA that builds tensor from sum of vectors
    aca_vects_norms = aca_t(big_t, max_rank, random_seed=random_seed)
    # aca_matrix_norms = aca2(big_t, max_rank, start_col=None, random_seed=random_seed)
    # aca_k_hat_norms = aca_tensor(big_t, max_rank, start_col=None, random_seed=random_seed)


    k_hat = 3
    # aca_k_hat_norms = aca_k_vectors(big_t, max_rank, k_hat=k_hat, start_tube=None, random_seed=random_seed)

    # Try out tube-matrix
    # aca_matrix_norms = matrix2(big_t, max_rank, start_col=None, random_seed=random_seed)

    # Matrix-Tube implementation
    aca_matrix_norms = aca_matrix_x_vector(big_t, max_rank, start_matrix=None, random_seed=random_seed)

    # cp_norms = []
    # for rank in range(1, max_rank+1):
    #     factors_cp = parafac(big_t, rank=rank, normalize_factors=False)
    #     cp_norm = compare_cp_with_full(cp=factors_cp, original=big_t)
    #     cp_norms.append(cp_norm)

    print("ACA vectors: ", aca_vects_norms)
    print("ACA k_hat:", aca_k_hat_norms)
    print("ACA matrix:", aca_matrix_norms)
    # print("CP:", cp_norms)

    if plot:
        plot_utils.plot_norms_aca_cp(max_rank, aca_v=aca_vects_norms, aca_k=aca_k_hat_norms, aca_m=aca_matrix_norms, cp_norms=cp_norms)
        # plotting.plot_amount_calcs(big_t, k_hat, max_rank)

def main():
    path = "tensors/full_tensor.npy"
    big_t = np.load(path)

    # big_t = random_tensor((5, 6, 6), 1, 20, seed=3)
    # print(big_t)

    print(f"Tensor is symmetric? -> {tensor_symmetric_test(big_t)}")
    compare_acas(big_t, max_rank=5, random_seed=2, plot=True)


if __name__ == "__main__":
    main()
