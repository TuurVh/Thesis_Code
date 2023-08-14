from ACA_implementations import aca_tensor, aca_k_vectors, aca_matrix_x_vector, compare_aca_original
from ACA_implementations import compare_cp_with_full
from ACA_T import aca_tensor as aca_vects
from ACA_T import get_CP_decomposition, get_norm
import plot_utils
import numpy as np

# Set print to print matrices and vectors in one line
np.set_printoptions(linewidth=np.inf)


def get_rel_error(tensor, max_rank, rds, method=''):
    if method == "method1" or method == "matrix":
        mats, m_ds, tubes = aca_matrix_x_vector(tensor, max_rank, random_seed=rds, to_cluster=True)
        err = compare_aca_original(mats, tubes, m_ds, tensor)
        return err

    if method == "method2" or method == "vectors":
        errs = aca_vects(tensor, max_rank, random_seed=rds)
        err = errs[max_rank-1]
        return err

    if method == "CP":
        cp = get_CP_decomposition(tensor, max_rank)
        err = get_norm(cp, tensor)
        return err


def main():
    path = "tensors/full_tensor.npy"
    big_t = np.load(path)
    shape = big_t.shape

    # big_t = random_tensor((5, 4, 4), 1, 6, seed=1)
    # print(big_t)
    max_rang = 50
    amount_iters = 10
    method = "CP"
    all_errs = []

    for r in range(5, max_rang+1, 5):
        print("for rank:", r)
        error_for_rank = []
        if method == 'CP':
            error_for_rank = get_rel_error(big_t, max_rank=r, method=method, rds=None)
            all_errs.append(error_for_rank)
        else:
            for s in range(amount_iters):
                seed = s
                error = get_rel_error(big_t, method=method, max_rank=r, rds=seed)
                error_for_rank.append(error)
            all_errs.append(error_for_rank)
    print("errs", all_errs)
    if method == 'CP':
        plot_utils.plot_rel_errs_cp(all_errs)
    else:
        plot_utils.plot_rel_errs(all_errs)


if __name__ == "__main__":
    main()
