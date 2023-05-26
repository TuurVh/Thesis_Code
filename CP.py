import numpy as np


import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.cluster import SpectralClustering
from utils import *

def create_matrix(size, zero_diag=False):
    # Initializing the matrix with all zeros
    matrix = [[0 for _ in range(size)] for _ in range(size)]

    # Filling the first 10 rows and columns with ones
    for i in range(size//2):
        for j in range(size//2):
            matrix[i][j] = 1

    # Filling the last 10 rows and columns with ones
    for i in range(size//2, size):
        for j in range(size//2, size):
            if i != j:
                matrix[i][j] = 1

    if zero_diag:
        for i in range(size):
            for j in range(size):
                matrix[i][j] = 0

    return np.array(matrix)


# Load a saved .npy tensor and calculate its rank.
# m = create_matrix(20, zero_diag=False)
# print(m)
tensor = np.load("tensors/full_tensor.npy")
rank = tensor.ndim
print("shape", tensor.shape, "rank", rank)

# Calculates the Parafac decomposition using the ALS algorithm.
factors = parafac(tensor, rank=3, normalize_factors=False)
# facts = parafac(m, rank=2)

print("parafac", factors[1])

# reconstruct the tensor from the factors
reconstructed_tensor = tl.cp_tensor.cp_to_tensor(factors)
# reconstructed_m = tl.cp_tensor.cp_to_tensor(facts)

print("tensor", tensor)
print("recon", reconstructed_tensor)
print("diff", np.rint(tensor-reconstructed_tensor))

# print("matirx", reconstructed_m)
# diff = m - reconstructed_m
# print("difference matrix", np.rint(diff))

# Apply spectral clustering to the new matrix
k = 3
print(factors[1][1])
# spectral_clustering = SpectralClustering(n_clusters=k).fit(factors[1][0])

# Label the columns of the factor matrices based on the resulting clustering
# labels = spectral_clustering.labels_
# print(labels)