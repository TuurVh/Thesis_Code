from ACA_implementations import aca_tensor, aca_k_vectors, aca_matrix_x_vector, compare_aca_original
from ACA_implementations import compare_cp_with_full
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
        cs, rs, ts, c_ds, r_ds, t_ds = aca_tensor(tensor, 16, random_seed=None, to_cluster=True)
        shape = tensor.shape
        m_tot = np.zeros(shape=(shape[2], shape[1]))
        for i in range(len(cs)):
            new_row = np.divide(rs[i], r_ds[i])
            matrix = np.outer(cs[i], new_row)
            m_tot += matrix
        return m_tot


def main():
    path = "tensors/person2&3-all_ex_75ts.npy"
    big_t = np.load(path)
    shape = big_t.shape

    # big_t = random_tensor((5, 4, 4), 1, 6, seed=1)
    # print(big_t)
    max_rang = 30
    amount_iters = 10
    method = "matrix"
    all_errs = []

    for r in range(5, max_rang+1, 5):
        print("for rank:", r)
        error_for_rank = []
        for s in range(amount_iters):
            seed = s
            error = get_rel_error(big_t, method=method, max_rank=r, rds=seed)
            error_for_rank.append(error)
        all_errs.append(error_for_rank)
    plotting.plot_aris(all_errs)


if __name__ == "__main__":
    main()
