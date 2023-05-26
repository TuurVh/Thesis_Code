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


def main():
    path = "tensors/person3-5-10_all_ex_50ts.npy"
    big_t = np.load(path)
    shape = big_t.shape

    # big_t = random_tensor((5, 4, 4), 1, 6, seed=1)
    # print(big_t)

    # Dimensie van personen
    # cs, rs, ts, c_ds, r_ds, t_ds = aca_tensor(big_t, 32, random_seed=None, to_cluster=True)
    # print(len(cs))

    # row = np.divide(rs[0], r_ds[0])
    # dist = np.outer(cs[0], row)
    # print("D", dist.shape)
    #
    # cols = np.vstack(cs)
    # temp = [np.divide(row, rd) for row, rd in zip(rs, r_ds)]
    # rows = (np.vstack(temp))
    #
    # d_matrix = np.dot(cols.T, rows)

    # Dimensie van sensoren
    cs, rs, ts, c_ds, r_ds, t_ds = aca_tensor(big_t, 16, random_seed=None, to_cluster=True)
    m_tot = np.zeros(shape=(shape[2], shape[1]))
    for i in range(len(cs)):
        new_row = np.divide(rs[i], r_ds[i])
        matrix = np.outer(cs[i], new_row)
        m_tot += matrix

    d_matrix = m_tot

    # # Get the maximum dimension
    # max_dim = max(d_matrix.shape)
    #
    # # Create a square matrix of zeros with the maximum dimension
    # square_matrix = np.zeros((max_dim, max_dim))
    #
    # # Insert the elements from the original matrix into the square matrix
    # square_matrix[:d_matrix.shape[0], :d_matrix.shape[1]] = d_matrix
    amount_clusters = 3
    res_spectral = clustering.spectral(d_matrix, amount_clusters)
    print(f"Result for spectral clustering with {amount_clusters} clusters: \n {res_spectral}")

    clusters, medoids = clustering.k_medoids(d_matrix, amount_clusters)
    print(f"Result for K-medoids with {amount_clusters} clusters:")
    for i, c in enumerate(clusters):
        print(f"Cluster {i}: {c}")
    print(f"with medoids: {medoids}")

    ground_truth = ([0]*6 + [1]*6 + [2]*6)*3
    ari = clustering.get_ARI_spectral(d_matrix, amount_clusters, ground_truth)
    print("ARI spectral =", ari)

    ari = clustering.get_ARI_k_medoids(d_matrix, amount_clusters, ground_truth)
    print("ARI k-medoids =", ari)


if __name__ == "__main__":
    main()
