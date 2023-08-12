import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from pyclustering.cluster.kmedoids import kmedoids
import random
import pandas as pd
import dtaidistance
from sklearn.metrics.cluster import adjusted_rand_score


def distance_to_similarity(D, r=None):
    """
    Based on similarity implementation from dtaidistance library:
    https://github.com/wannesm/dtaidistance/tree/master
    Transform a distance matrix to a similarity matrix
    :param D: Distance matrix
    :param r: Scaling or smoothing parameter
    :return: S, the similarity matrix
    """
    if r is None:
        r = np.max(D)
    S = np.exp(-D / r)
    return S


def make_symmetric(S):
    return (S + S.T) / 2


def k_means(vectors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(vectors)
    return kmeans


def get_ARI_k_means(kmeans):
    real_classes = get_GT()
    print("real  :", real_classes)
    print("approx:", ', '.join(map(str, kmeans.labels_)))
    score = adjusted_rand_score(real_classes, kmeans.labels_)
    return score


def get_GT():
    data = pd.read_csv('overview.csv')
    exercise_mapping = {'squat': 0, 'lunge': 1, 'sidelunge': 2}
    exercise_list = [exercise_mapping[exercise] for exercise in data['exercise']]

    return exercise_list


def spectral(distance_matrix, n_clusters):
    similarity_matrix = distance_to_similarity(distance_matrix)
    symmetric_sim_matrix = make_symmetric(similarity_matrix)
    spectral_model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    result_spectral = spectral_model.fit_predict(symmetric_sim_matrix)
    return result_spectral


def get_ARI_spectral(matrix, n_clusters, real_classes):
    results = spectral(matrix, n_clusters)
    score = adjusted_rand_score(real_classes, results)
    return score


def k_medoids(distance_matrix, n_clusters, initial_medoids=None):
    symmetric_dist_matrix = make_symmetric(distance_matrix)
    if initial_medoids is None:
        initial_medoids = random.sample(range(distance_matrix.shape[0]), k=n_clusters)
    kmedoids_instance = kmedoids(symmetric_dist_matrix, initial_medoids, data_type='distance_matrix')
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    centers = kmedoids_instance.get_medoids()
    return clusters, centers


def get_ARI_k_medoids(matrix, n_clusters, real_classes):
    clusters, _ = k_medoids(matrix, n_clusters)
    length = matrix.shape[0]
    results = transform_medoids_outcome(clusters, length)
    score = adjusted_rand_score(real_classes, results)
    return score


def transform_medoids_outcome(clusters, l):
    res = np.zeros(l)
    cluster_indication = 0
    for cluster in clusters:
        for element in cluster:
            res[element] = cluster_indication
        cluster_indication += 1
    print(res)
    return res
