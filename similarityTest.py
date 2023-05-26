import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score as ars
from ftsc.data_loader import load_timeseries_from_multiple_tsvs
from ftsc.cluster_problem import ClusterProblem
from ftsc.aca import aca_symm
from tests.tests_utils import get_labels, get_amount_of_classes
from similarities import distance_to_similarity, squash
from ftsc.solradm import solradm
import matplotlib.pyplot as plt

dataset = "CBF"
train_path = "tests/Data/" + dataset + "/" + dataset + "_TRAIN.tsv"
test_path = "tests/Data/" + dataset + "/" + dataset + "_TEST.tsv"

labels, series = load_timeseries_from_multiple_tsvs(train_path, test_path)


def spectral(matrix, class_nb):
    model_agglo = SpectralClustering(n_clusters=class_nb, affinity='precomputed', random_state=0)
    result_agglo = model_agglo.fit_predict(matrix)
    return result_agglo


def compare_spectral(matrix, k, real_classes, compare_func=ars):
    results = spectral(matrix, k)
    score = compare_func(real_classes, results)
    return score


def compare_spectral_sigm(matrix, k, real_classes, compare_func=ars):
    model_agglo = SpectralClustering(n_clusters=k, affinity='precomputed_nearest_neighbors', random_state=0)
    results = model_agglo.fit_predict(matrix)
    score = compare_func(real_classes, results)
    return score


def scatter(matrix, title):
    # flat = []
    # for i in range(0, len(matrix)):
    #     for j in range(i+1, len(matrix)):
    #         flat.append(matrix[i][j])

    flat = np.matrix.flatten(matrix)

    x = range(0, len(flat))
    plt.scatter(x, flat, s=0.01)
    plt.title(title)
    plt.show()


def print_metrics(matrix):
    print("Min: ", np.min(matrix))
    print("Mean: ", np.mean(matrix))


def solrad_dtw():
    cp = ClusterProblem(series, "dtw")

    dist = solradm(cp, 50, epsilon=2.0)
    print("D,mean, max:", np.mean(dist), np.max(dist))
    full_dist = cp.sample_full_matrix()
    full_sim = distance_to_similarity(full_dist)

    # transform distance matrix to similarity matrix with all possible transformations, r is still a parameter
    sim_exp = distance_to_similarity(dist, r=None, method='exponential')
    print_metrics(sim_exp)
    # scatter(sim_exp, 'exponential')
    sim_gauss = distance_to_similarity(dist, r=None, method='gaussian')
    print_metrics(sim_gauss)
    # scatter(sim_gauss, 'gauss')
    sim_recip = distance_to_similarity(dist, r=None, method='reciprocal')
    print_metrics(sim_recip)
    # scatter(sim_recip, 'reciprocal')
    sim_rev = distance_to_similarity(dist, r=None, method='reverse')
    print_metrics(sim_rev)
    # scatter(sim_rev, 'reverse')

    sigm_log = squash(dist, x0=np.mean(dist), method='logistic')
    print_metrics(sigm_log)
    # scatter(sigm_log, 'sigmoid log')

    sigm_sim = squash(sim_exp, x0=np.mean(sim_exp), method='eigen')
    print_metrics(sigm_sim)
    # scatter(sigm_sim, 'sigmoid sim')

    # get the labeled set and amount of labels (respect. for ARI and clustering)
    ground_truth = get_labels(dataset)
    class_nb = get_amount_of_classes(ground_truth)

    score1 = compare_spectral(full_sim, class_nb, ground_truth)
    print("ARI FULL: ", score1)

    score2 = compare_spectral(sim_exp, class_nb, ground_truth)
    print("ARI exponential: ", score2)
    score3 = compare_spectral(sim_gauss, class_nb, ground_truth)
    print("ARI gaussian: ", score3)
    score4 = compare_spectral(sim_recip, class_nb, ground_truth)
    print("ARI reciprocal: ", score4)
    score5 = compare_spectral(sim_rev, class_nb, ground_truth)
    print("ARI reverse: ", score5)

    score6 = compare_spectral_sigm(sigm_log, class_nb, ground_truth)
    print("ARI sigmoid: ", score6)

    score = compare_spectral(sigm_sim, class_nb, ground_truth)
    print("ARI sigmoid sim: ", score)
    return score1, score2, score3, score4, score5, score6


def dtw_sims():
    cp = ClusterProblem(series, "dtw")

    dist = aca_symm(cp, tolerance=0.05, max_rank=20)
    print("D,mean, max:", np.mean(dist), np.max(dist))
    full_dist = cp.sample_full_matrix()
    full_sim = distance_to_similarity(full_dist, method='gaussian')

    # transform distance matrix to similarity matrix with all possible transformations, r is still a parameter
    sim_exp = distance_to_similarity(dist, r=None, method='exponential')
    print_metrics(sim_exp)
    # scatter(sim_exp, 'exponential')
    sim_gauss = distance_to_similarity(dist, r=None, method='gaussian')
    print_metrics(sim_gauss)
    # scatter(sim_gauss, 'gauss')
    sim_recip = distance_to_similarity(dist, r=None, method='reciprocal')
    print_metrics(sim_recip)
    # scatter(sim_recip, 'reciprocal')
    sim_rev = distance_to_similarity(dist, r=None, method='reverse')
    print_metrics(sim_rev)
    # scatter(sim_rev, 'reverse')

    sigm_log = squash(dist, x0=np.mean(dist), method='logistic')
    print_metrics(sigm_log)
    # scatter(sigm_log, 'sigmoid log')

    sigm_sim = squash(sim_exp, x0=np.mean(sim_exp), method='eigen')
    print_metrics(sigm_sim)
    # scatter(sigm_sim, 'sigmoid sim')

    # get the labeled set and amount of labels (respect. for ARI and clustering)
    ground_truth = get_labels(dataset)
    class_nb = get_amount_of_classes(ground_truth)

    score1 = compare_spectral(full_sim, class_nb, ground_truth)
    print("ARI FULL: ", score1)

    score2 = compare_spectral(sim_exp, class_nb, ground_truth)
    print("ARI exponential: ", score2)
    score3 = compare_spectral(sim_gauss, class_nb, ground_truth)
    print("ARI gaussian: ", score3)
    score4 = compare_spectral(sim_recip, class_nb, ground_truth)
    print("ARI reciprocal: ", score4)
    score5 = compare_spectral(sim_rev, class_nb, ground_truth)
    print("ARI reverse: ", score5)

    score6 = compare_spectral_sigm(sigm_log, class_nb, ground_truth)
    print("ARI sigmoid: ", score6)

    score = compare_spectral(sigm_sim, class_nb, ground_truth)
    print("ARI sigmoid sim: ", score)
    return score1, score2, score3, score4, score5, score6


def msm_sims():
    return


def ari_plot(full, exp, gauss, recip, rev, sigm):
    x = range(len(full))
    plt.xticks(range(0, 5))
    plt.xlabel("iteration")
    plt.ylabel("ARI-score")
    plt.plot(x, full, marker='.'); plt.plot(x, exp, marker='.'); plt.plot(x, gauss, marker='.')
    plt.plot(x, recip, marker='.'); plt.plot(x, rev, marker='.');
    plt.legend(['Full matrix', 'Exponential S', 'Gaussian S', 'Reciprocal S', 'Reverted S'])
    plt.show()


def main():
    full = []; exp = []; gauss = []; recip = []; rev = []; sigm = []
    for i in range(0, 5):
        f, e, g, rc, rv, s = dtw_sims()
        full.append(f); exp.append(e); gauss.append(g); recip.append(rc); rev.append(rv); sigm.append(s)
    ari_plot(full, exp, gauss, recip, rev, sigm)

    # full = []; exp = []; gauss = []; recip = []; rev = []; sigm = []
    # for i in range(0, 5):
    #     f, e, g, rc, rv, s = solrad_dtw()
    #     full.append(f); exp.append(e); gauss.append(g); recip.append(rc); rev.append(rv); sigm.append(s)
    # ari_plot(full, exp, gauss, recip, rev, sigm)

    # dtw_sims()
    # solrad_dtw()


if __name__ == '__main__':
    main()
