from ACA_implementations import aca_matrix_x_vector
from ACA_T import aca_tensor, get_CP_decomposition
from ACA_implementations import compare_cp_with_full
from cluster_utils import *
import plot_utils
import numpy as np

# Set print to print matrices and vectors in one line
np.set_printoptions(linewidth=np.inf)

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

    amount_clusters = 3
    ground_truth = ([0] * 6 + [1] * 6 + [2] * 6) * 3

    for r in range(5, max_rang+1, 5):
        print("for rank:", r)
        aris_spect = []
        aris_medoid = []
        for s in range(amount_iters):
            seed = s
            d_matrix = get_dist_matrix(big_t, method=method, max_rank=r, rds=seed)

            ari_s = get_ARI_spectral(d_matrix, amount_clusters, ground_truth)
            print("ARI spectral =", ari_s)
            aris_spect.append(ari_s)

            ari_m = get_ARI_k_medoids(d_matrix, amount_clusters, ground_truth)
            print("ARI k-medoids =", ari_m)
            aris_medoid.append(ari_m)
        all_spect.append(aris_spect)
        all_medoid.append(aris_medoid)
    print("res,", all_spect)
    plot_utils.plot_aris(all_spect, all_medoid)


def ARI_kmeans_CP(tensor, max_rang, amount_iters, k, feature_vect='rows'):
    all_kmeans = []
    all_CP = []
    for r in range(5, max_rang + 1, 10):
        print("for rank:", r)
        aris_kmeans = []
        cp_facts = get_CP_decomposition(tensor, r)[1]
        if feature_vect == 'rows':
            to_cluster = cp_facts[2]
        else:
            to_cluster = cp_facts[0]
        cp_km = k_means(to_cluster, n_clusters=k, store_result=False)
        ari_cp = get_ARI_k_means(cp_km, fv=feature_vect)
        all_CP.append(ari_cp)
        for _ in range(amount_iters):
            rs, cs, ts, r_ds, c_ds = aca_tensor(tensor, r, random_seed=None, to_cluster=True)
            if feature_vect == 'rows':
                to_cluster = np.transpose(rs)
            else:
                to_cluster = np.transpose(ts)
            km = k_means(to_cluster, n_clusters=k, store_result=False)
            ari = get_ARI_k_means(km, fv=feature_vect)
            aris_kmeans.append(ari)
        all_kmeans.append(aris_kmeans)
    plot_utils.plot_kmeans_aris(all_kmeans, all_CP)


def get_k_means(big_t, method, max_rank):
    if method == "method2":
        rs, cs, ts, r_ds, c_ds = aca_tensor(big_t, max_rank, random_seed=0, to_cluster=True)
        to_cluster = np.transpose(rs)
        print(to_cluster)
        # to_cluster = np.transpose(cs)
    elif method == "CP":
        factors = get_CP_decomposition(big_t, max_rank)[1]
        to_cluster = factors[2]
        print("facts", factors)
    km = k_means(to_cluster, n_clusters=3, store_result=False)
    ari = get_ARI_k_means(km)
    print("de ari score is", ari)


def main():
    path = "tensors/full_tensor.npy"
    big_t = np.load(path)
    k = 6
    max_rang = 45
    amount_iters = 10
    # Method can be "method2" or "CP"
    method = "method2"
    # spect_and_medoids(big_t, method, max_rang, amount_iters)
    # get_k_means(big_t, method, max_rang)
    ARI_kmeans_CP(big_t, max_rang, amount_iters, k, feature_vect='rows')


if __name__ == "__main__":
    main()
