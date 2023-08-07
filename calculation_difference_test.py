from ACA_implementations import aca_tensor, aca_k_vectors, aca_matrix_x_vector, compare_cp_with_full, \
    compare_aca_original
import plotting
import numpy as np

def amount_calculations(big_t, max_rank, random_seed, method=''):
    shape = big_t.shape
    x = shape[2]
    y = shape[1]
    z = shape[0]
    if method == "matrix" or method == "method1":
        mats, m_ds, tubes = aca_matrix_x_vector(big_t, max_rank, start_matrix=None, random_seed=random_seed, to_cluster=True)
        err = compare_aca_original(mats, tubes, m_ds, big_t)
        amount_calcs = (((x*(x+1))/2) + z) * max_rank
        percentage = (((x*(x+1))/2) * z)
        return err, amount_calcs, percentage
    elif method == "k_hat" or method == "method3":
        return 0
    return 0


def main():
    path = "tensors/person2&3-all_ex_75ts.npy"
    tensor = np.load(path)

    max_rang = 40
    amount_iters = 50
    method = "matrix"
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