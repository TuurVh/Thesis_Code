import numpy as np
from sklearn.cluster import SpectralClustering
from similarities import distance_to_similarity, squash
from ftsc.data_loader import load_timeseries_from_multiple_tsvs
from ftsc.cluster_problem import ClusterProblem
from ftsc.aca import aca_symm
from sklearn.metrics import adjusted_rand_score as ars
import time

from tests.tests_utils import get_labels, get_amount_of_classes

dataset = "CBF"
train_path = "tests/Data/" + dataset + "/" + dataset + "_TRAIN.tsv"
test_path = "tests/Data/" + dataset + "/" + dataset + "_TEST.tsv"

labels, series = load_timeseries_from_multiple_tsvs(train_path, test_path)
amount_its = 5


def distance_cp(cp):
    """amount_its amount of iterations of approximated and exact matrices (DISTANCE)"""
    start_time = time.time()
    approx = aca_symm(cp, tolerance=0.05, max_rank=20)
    end_time = time.time()
    print("Time spent on approximation with ACA: " + str(end_time - start_time) + " seconds")

    start_time = time.time()
    full = cp.sample_full_matrix()
    end_time = time.time()
    print("Time spent on full matrix: " + str(end_time - start_time) + " seconds")

    dist_rel_err = cp.get_relative_error(approx)

    return approx, full, dist_rel_err


def similarity_cp(cp):
    """amount_its amount of iterations of approximated and exact matrices (SIMILARITY)"""
    start_time = time.time()
    approx = aca_symm(cp, tolerance=0.05, max_rank=20)
    end_time = time.time()
    # transform distance matrix to similarity, with parameter r and method = exp, gauss, reciprocal of reverse
    # r = None
    # approx_sim = distance_to_similarity(approx, r, 'exponential')

    # print("Time spent on approximation with ACA: " + str(end_time - start_time) + " seconds")
    return approx


def dist_sim_full():
    cp = ClusterProblem(series, "dtw")
    approx = aca_symm(cp, tolerance=0.05, max_rank=20)
    approx_sim = distance_to_similarity(approx, None, 'exponential')
    exact = cp.sample_full_matrix()
    return approx, approx_sim, exact


def spectral(matrix, class_nb):
    model_agglo = SpectralClustering(n_clusters=class_nb, affinity='precomputed', assign_labels='discretize', random_state=0)
    result_agglo = model_agglo.fit_predict(matrix)
    return result_agglo


def compare_spectral(matrix, k, real_classes, compare_func=ars):
    results = spectral(matrix, k)
    print(np.subtract(real_classes, results))
    score = compare_func(real_classes, results)
    return score


def main():
    cp = ClusterProblem(series, "msm")

    dist, dist_full, dist_rel_err = distance_cp(cp)

    # Transform solved dist to sim matrix, update this in CP and then use SIM matrix ipv DIST for approx
    sim_full = distance_to_similarity(dist_full)
    sim = distance_to_similarity(dist)

    cp.set_solved_matrix(sim_full)
    sim_rel_err = cp.get_relative_error(sim)

    sigm_full = squash(dist_full, x0=np.mean(dist_full))
    sigm = squash(dist, x0=np.mean(dist))
    cp.set_solved_matrix(sigm_full)
    sigm_rel_err = cp.get_relative_error(sigm)

    # print("------- DISTANCE MATRIX -------")
    # print(dist)
    # print("------- EXACT MATRIX -------")
    # print(dist_full)
    # print("RELATIVE ERROR DISTANCE = ", dist_rel_err)
    # #
    # print("------- SIMILARITY MATRIX -------")
    # print(sim)
    # print("------- SIMILARITY MATRIX 2 -------")
    # print(sim2)
    # print("------- FULL SIM MATRIX -------")
    # print(sim_full)
    # print("RELATIVE ERROR SIM = ", sim_rel_err)
    # print("RELATIVE ERROR SIM2 = ", sim_rel_err2)

    # print("------- SiGmOiD FULL MATRIX -------")
    # print(sigm_full)
    # print(np.mean(sigm_full))
    # print("------- SiGmOiD MATRIX -------")
    # print(sigm)
    # print("RELATIVE ERROR SIGM = ", sigm_rel_err)

    ground_truth = get_labels(dataset)
    class_nb = get_amount_of_classes(ground_truth)

    score = compare_spectral(sim, class_nb, ground_truth)
    print("ARI: ", score)


if __name__ == "__main__":
    main()

def aca_k_vectors(tensor, max_rank, k_hat, start_col=None, random_seed=None):
    print("shape", tensor.shape)

    rows = []
    cols = []
    tubes = []
    r_deltas = []
    c_deltas = []
    t_deltas = []

    xs_used = []
    ys_used = []
    z_used = []
    aca_k_hat_norms = []

    # Def shape of tensor
    shape = tensor.shape  # shape = (z, y, x)

    # If a random seed is given, use this.
    if random_seed is not None:
        random.seed(random_seed)

    # If no start column is given, initialize a random one.
    if start_col is None:
        x_as = random.randint(0, shape[2]-1)
        z_as = random.randint(0, shape[0]-1)
        print(f"chosen col: x={x_as}, z={z_as}")

    else:
        x_as = start_col[0]
        z_as = start_col[1]

    xs = [x_as - i for i in range(k_hat)]
    xs.reverse()
    m_row_idx = 0
    xs_used.append(xs)
    z_used.append(z_as)

    rank = 0
    # Select new skeletons until desired rank is reached.
    while rank < max_rank:

        print(f"--------------------- RANK {rank} ----------------------")
        # print(tensor[z_as])

        # --------- COLUMNS ---------
        col_fibers = tensor[z_as, :, xs]
        print("col before", col_fibers)

        approx = np.zeros(col_fibers.shape)
        for i in range(rank):
            approx = np.add(approx, cols[i] * rows[i][m_row_idx] * tubes[i][z_as] * (1.0 / r_deltas[i]) * (1.0 / t_deltas[i]))
        new_cols = np.subtract(col_fibers, approx)
        print("col after", new_cols)

        new_abs_cols = abs(new_cols)
        m_col_idx = np.unravel_index(np.argmax(new_abs_cols), new_cols.shape)
        max_val = col_fibers[m_col_idx]

        print(f"max val: {max_val} on Y pos: {m_col_idx}")
        y_as = m_col_idx[1]
        cols.append(new_cols)
        c_deltas.append(max_val)

        ys = [y_as - i for i in range(k_hat)]
        ys.reverse()
        ys_used.append(ys)

        # --------- ROWS ---------
        row_fibers = tensor[z_as, ys, :]
        print("r before", row_fibers)

        approx = np.zeros(row_fibers.shape)
        for i in range(rank):
            approx = np.add(approx, cols[i][m_col_idx] * rows[i] * tubes[i][z_as] * (1.0 / c_deltas[i]) * (1.0 / t_deltas[i]))
        new_rows = np.subtract(row_fibers, approx)
        print("r after", new_rows)

        new_abs_rows = abs(new_rows)
        m_row_idx = np.unravel_index(np.argmax(new_abs_rows), new_rows.shape)
        max_val = row_fibers[m_row_idx]

        print(f"max val: {max_val} on X pos: {m_row_idx}")
        x_as = m_row_idx[1]
        xs = [x_as - i for i in range(k_hat)]
        xs.reverse()
        rows.append(new_rows)
        r_deltas.append(max_val)
        xs_used.append(xs)

        # --------- TUBES ---------
        tube_fiber = tensor[:, y_as, x_as]
        print("t:", tube_fiber)

        approx = np.zeros(len(tube_fiber))
        for i in range(rank):
            approx = np.add(approx, cols[i][m_col_idx] * rows[i][m_row_idx] * tubes[i] * (1.0 / c_deltas[i]) * (1.0 / r_deltas[i]))
        new_tube = np.subtract(tube_fiber, approx)
        print("t after:", new_tube)

        new_abs_tube = abs(new_tube)
        tube_without_previous = np.delete(new_abs_tube, z_used, axis=0)
        max_val = np.max(tube_without_previous)
        z_as = np.where(new_abs_tube == max_val)[0][0]
        print(f"max val: {new_tube[z_as]} on Z pos: {z_as}")

        tubes.append(new_tube)
        t_deltas.append(new_tube[z_as])
        z_used.append(z_as)

        matrices = []
        for i in range(len(cols)):
            col = cols[i]
            print("col", col)
            print("row", rows[i])
            print("delta", r_deltas[i])
            row = np.divide(rows[i], r_deltas[i])
            matrices.append(np.dot(col.T, row))

        aca_norm = compare_aca_original(matrices, tubes, t_deltas, tensor)
        aca_k_hat_norms.append(aca_norm)

        rank += 1
    return aca_k_hat_norms
