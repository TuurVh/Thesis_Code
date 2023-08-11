from ACA_implementations import aca_k_vectors, aca_matrix_x_vector, compare_cp_with_full, \
    compare_aca_original
from ACA_T import aca_tensor
import plotting
import numpy as np

def amount_calculations(big_t, max_rank, random_seed, method=''):
    shape = big_t.shape
    x = shape[2]
    y = shape[1]
    z = shape[0]
    percentage = (((x*(x+1))/2) * z)
    if method == "matrix" or method == "method1":
        mats, m_ds, tubes = aca_matrix_x_vector(big_t, max_rank, start_matrix=None, random_seed=random_seed, to_cluster=True)
        err = compare_aca_original(mats, tubes, m_ds, big_t)
        amount_calcs = (((x*(x+1))/2) + z) * max_rank
        return err, amount_calcs, percentage
    elif method == "vectors" or method == "method2":
        errs = aca_tensor(big_t, max_rank, random_seed=random_seed)
        err = errs[max_rank-1]
        amount_calcs = ((x + y) + z) * max_rank
        return err, amount_calcs, percentage
    return 0


def main():
    path = "tensors/person2&3-all_ex_75ts.npy"
    tensor = np.load(path)

    max_rang = 50
    amount_iters = 2
    method = "method2"
    all_errs = []
    all_calcs = []
    p = 1

    for r in range(5, max_rang+1, 5):
        print("for rank:", r)
        calcs = []
        for s in range(amount_iters):
            err, calc, p = amount_calculations(tensor, r, random_seed=s, method=method)
            calcs.append(calc)
        all_errs.append(err)
        calc = np.mean(calcs)
        all_calcs.append(calc)
    print(all_errs, all_calcs)
    plotting.plot_err_to_calcs(all_errs, all_calcs)
    plotting.plot_err_to_calcs_percentage(all_errs, all_calcs, p)


if __name__ == "__main__":
    main()