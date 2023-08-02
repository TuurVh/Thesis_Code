from ACA_implementations import aca_tensor, aca_k_vectors, aca_matrix_x_vector
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


def get_dist_matrix(tensor, aca_matrix=False, aca_vects=False, aca_k_hat=False):
    if aca_matrix:
        mats, m_ds = aca_matrix_x_vector(tensor, 10, to_cluster=True)
        m_tot = np.zeros(shape=mats[0].shape)
        for i in range(len(mats)):
            mat = np.divide(mats[i], m_ds[i])
            m_tot += mat
        return m_tot

    if aca_vects:
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

    # big_t = random_tensor((5, 4, 4), 1, 6, seed=1)
    # print(big_t)

    d_matrix = get_dist_matrix(big_t, aca_matrix=True)
    amount_clusters = 3
    res_spectral = clustering.spectral(d_matrix, amount_clusters)
    print(f"Result for spectral clustering with {amount_clusters} clusters: \n {res_spectral}")

    clusters, medoids = clustering.k_medoids(d_matrix, amount_clusters)
    print(f"Result for K-medoids with {amount_clusters} clusters:")
    for i, c in enumerate(clusters):
        print(f"Cluster {i}: {c}")
    print(f"with medoids: {medoids}")

    ground_truth = ([0]*6 + [1]*6 + [2]*6)*2
    ari = clustering.get_ARI_spectral(d_matrix, amount_clusters, ground_truth)
    print("ARI spectral =", ari)

    ari = clustering.get_ARI_k_medoids(d_matrix, amount_clusters, ground_truth)
    print("ARI k-medoids =", ari)


if __name__ == "__main__":
    main()
