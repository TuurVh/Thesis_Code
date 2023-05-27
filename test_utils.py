import numpy as np

def matrix_symmetric_test(matrix):
    # matrix is symmetric if elements are same as the transpose
    return np.allclose(matrix, matrix.T)


def tensor_symmetric_test(tensor):
    results = []
    for m in tensor:
        results.append(matrix_symmetric_test(m))

    return all(results)
