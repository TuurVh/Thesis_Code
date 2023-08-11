from ACA_implementations import aca_tensor, aca_k_vectors, aca_matrix_x_vector, compare_aca_original
from ACA_implementations import compare_cp_with_full
from ACA_T import aca_tensor as aca_vects
import clustering
import plotting
from tensorly.decomposition import parafac
import numpy as np

# Set print to print matrices and vectors in one line
np.set_printoptions(linewidth=np.inf)


def random_tensor(shape, low, high, seed):
    np.random.seed(seed)
    tensor = np.random.randint(low=low, high=high, size=shape)
    for i in range(shape[0]):
        np.fill_diagonal(tensor[i], 0)
    return tensor


def get_rel_error(tensor, max_rank, rds, method=''):
    if method == "method1" or method == "matrix":
        mats, m_ds, tubes = aca_matrix_x_vector(tensor, max_rank, random_seed=rds, to_cluster=True)
        err = compare_aca_original(mats, tubes, m_ds, tensor)

        return err

    if method == "method2" or method == "vectors":
        errs = aca_vects(tensor, max_rank, random_seed=rds)
        err = errs[max_rank-1]
        return err


def main():
    path = "tensors/person2&3-all_ex_75ts.npy"
    big_t = np.load(path)
    shape = big_t.shape

    # big_t = random_tensor((5, 4, 4), 1, 6, seed=1)
    # print(big_t)
    max_rang = 20
    amount_iters = 20
    method = "method2"
    all_errs = []

    for r in range(5, max_rang+1, 5):
        print("for rank:", r)
        error_for_rank = []
        for s in range(amount_iters):
            seed = s
            error = get_rel_error(big_t, method=method, max_rank=r, rds=seed)
            error_for_rank.append(error)
        all_errs.append(error_for_rank)
    print("errs", all_errs)
    plotting.plot_rel_errs(all_errs)


if __name__ == "__main__":
    main()
