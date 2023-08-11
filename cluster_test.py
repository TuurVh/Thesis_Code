from ACA_implementations import aca_matrix_x_vector
from ACA_T import aca_tensor
from ACA_implementations import compare_cp_with_full
from clustering import *
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


def get_dist_matrix(tensor, max_rank, rds, method=''):
    if method == "method1" or method == "matrix":
        mats, m_ds, _ = aca_matrix_x_vector(tensor, max_rank, random_seed=rds, to_cluster=True)
        m_tot = np.zeros(shape=mats[0].shape)
        for i in range(len(mats)):
            mat = np.divide(mats[i], m_ds[i])
            m_tot += mat
        return m_tot

    if method == "method2" or method == "vectors":
        rs, cs, ts, r_ds, c_ds = aca_tensor(tensor, max_rank, random_seed=None, to_cluster=True)
        shape = tensor.shape
        m_tot = np.zeros(shape=(shape[2], shape[1]))
        for i in range(len(cs)):
            new_row = np.divide(rs[i], r_ds[i])
            matrix = np.outer(cs[i], new_row)
            m_tot += matrix
        return m_tot


def spect_and_medoids(big_t, method, max_rang, amount_iters):
    all_spect = []
    all_medoid = []

    for r in range(5, max_rang+1, 5):
        print("for rank:", r)
        aris_spect = []
        aris_medoid = []
        for s in range(amount_iters):
            seed = s
            d_matrix = get_dist_matrix(big_t, method=method, max_rank=r, rds=seed)
            amount_clusters = 3
            # res_spectral = clustering.spectral(d_matrix, amount_clusters)
            # print(f"Result for spectral clustering with {amount_clusters} clusters: \n {res_spectral}")

            # clusters, medoids = clustering.k_medoids(d_matrix, amount_clusters)
            # print(f"Result for K-medoids with {amount_clusters} clusters:")
            # for i, c in enumerate(clusters):
            #     print(f"Cluster {i}: {c}")
            # print(f"with medoids: {medoids}")

            ground_truth = ([0]*6 + [1]*6 + [2]*6)*3
            ari_s = get_ARI_spectral(d_matrix, amount_clusters, ground_truth)
            print("ARI spectral =", ari_s)
            aris_spect.append(ari_s)

            ari_m = get_ARI_k_medoids(d_matrix, amount_clusters, ground_truth)
            print("ARI k-medoids =", ari_m)
            aris_medoid.append(ari_m)
        all_spect.append(aris_spect)
        all_medoid.append(aris_medoid)
    print("res,", all_spect)
    plotting.plot_aris(all_spect, all_medoid)


def get_k_means(big_t, method, max_rank, amount_iters):
    rs, cs, ts, r_ds, c_ds = aca_tensor(big_t, max_rank, random_seed=None, to_cluster=True)
    k_means(rs, n_clusters=3)


def main():
    path = "tensors/person2&3-all_ex_75ts.npy"
    big_t = np.load(path)
    shape = big_t.shape

    # big_t = random_tensor((5, 4, 4), 1, 6, seed=1)
    # print(big_t)
    max_rang = 30
    amount_iters = 10
    method = "method2"
    # spect_and_medoids(big_t, method, max_rang, amount_iters)
    get_k_means(big_t, method, max_rang, amount_iters)


if __name__ == "__main__":
    main()
