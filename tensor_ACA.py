import random
import numpy as np
import tensorly as tl
import logging
import plotting
from math import sqrt

# Create logger
logger = logging.getLogger("log_ACA")

# Set print to print matrices and vectors in one line
np.set_printoptions(linewidth=np.inf)

def aca_tensor(tensor, max_rank, start_col=None, random_seed=None, to_cluster=False):
    rows = []
    cols = []
    tubes = []

    r_deltas = []
    c_deltas = []
    t_deltas = []

    Is = []
    Js = []
    Ks = []

    aca_vects_norms = []

    # If a random seed is given, use this.
    if random_seed is not None:
        random.seed(random_seed)

    # Generate random samples for stopping criteria
    sample_indices, sample_values = generate_samples(tensor, max_rank)
    print(f"Sample indices: {sample_indices} \n with sample values {sample_values}")
    sample_size = len(sample_values)

    # Def shape of tensor
    shape = tensor.shape  # shape = (z, y, x)
    print(shape)

    # If no start column is given, initialize a random one.
    if start_col is None:
        max_sample = np.max(np.abs(sample_values))
        index_sample_max = np.where(np.abs(sample_values) == max_sample)[0][0]

        j_val = sample_indices[index_sample_max, 1]
        k_val = sample_indices[index_sample_max, 2]
        # print(f"chosen col: j={j_val}, k={k_val} = {get_fiber(j=j_val, k=k_val)}")

    else:
        j_val = start_col[0]
        k_val = start_col[1]

    Js.append(j_val)
    Ks.append(k_val)

    rank = 1
    restartable_samples = sample_values
    restartable_indices = sample_indices
    while rank < max_rank:
        logger.debug(f"----------- RANK {rank} -----------")

        # COLUMN

        col_fiber = get_fiber(j=j_val, k=k_val)

        approx = np.zeros(len(col_fiber))
        for r in range(1, rank):
            prev_col = cols[r-1] * rows[r-1][j_val] * tubes[r-1][k_val] * (1/c_deltas[r-1]) * (1/r_deltas[r-1])
            approx = np.add(approx, prev_col)

        new_col = np.subtract(col_fiber, approx)
        print(f"new col on k={k_val}, j={j_val} : ({col_fiber} - {approx}) =", new_col)

        while all(v == 0 for v in new_col):
            max_val = np.where(np.abs(restartable_samples) == max_residu)[0][0]
            j_val = restartable_indices[max_val][1]
            k_val = restartable_indices[max_val][0]

            col_fiber = get_fiber(j=j_val, k=k_val)

            approx = np.zeros(len(col_fiber))
            for r in range(1, rank):
                prev_col = cols[r - 1] * rows[r - 1][j_val] * tubes[r - 1][k_val] * (1 / r_deltas[r - 1]) * (
                            1 / t_deltas[r - 1])
                approx = np.add(approx, prev_col)
            new_col = np.subtract(col_fiber, approx)
            print(f"new col is now on k={k_val}, j={j_val} : ({col_fiber} - {approx}) =", new_col)

        cols.append(new_col)
        # update i
        # n_col = set_to_zero([j_val], new_col.copy())
        max_val, i_val = find_largest_absolute_value(new_col)
        print(f"The max value is {max_val} on position {i_val}\n")

        # if max_val == 0.0:
        #     max_val = np.where(np.abs(restartable_samples) == max_residu)[0][0]
        #     i_val = restartable_indices[max_val][2]
        #     print(f"The max value is now {max_val} on position {i_val}\n")

        c_deltas.append(max_val)
        Is.append(i_val)

        # ROW

        row_fiber = get_fiber(i=i_val, k=k_val)

        approx = np.zeros(len(row_fiber))
        for r in range(1, rank):
            prev_row = cols[r-1][i_val] * rows[r-1] * tubes[r-1][k_val] * (1/c_deltas[r-1]) * (1/r_deltas[r-1])
            approx = np.add(approx, prev_row)

        new_row = np.subtract(row_fiber, approx)
        print(f"new row on k={k_val}, i={i_val} : ({row_fiber} - {approx}) =", new_row)
        rows.append(new_row)

        # update j
        max_val, j_val = find_largest_absolute_value(new_row)
        print(f"The max value is {max_val} on position {j_val}\n")

        if max_val == 0.0:
            max_val = np.where(np.abs(restartable_samples) == max_residu)[0][0]
            j_val = restartable_indices[max_val][1]
            print(f"The max value is now {max_val} on position {j_val}\n")

        r_deltas.append(max_val)
        Js.append(j_val)

        # TUBE
        temp_j_val = Js[rank-1]
        tube_fiber = get_fiber(i=i_val, j=temp_j_val)

        approx = np.zeros(len(tube_fiber))
        for r in range(1, rank):
            prev_tube = cols[r-1][i_val] * rows[r-1][temp_j_val] * tubes[r-1] * (1/c_deltas[r-1]) * (1/r_deltas[r-1])
            approx = np.add(approx, prev_tube)

        new_tube = np.subtract(tube_fiber, approx)
        print(f"new tube on j={temp_j_val}, i={i_val} : ({tube_fiber} - {approx}) =", new_tube)
        tubes.append(new_tube)
        # update k
        max_val, k_val = find_largest_absolute_value(new_tube)
        print(f"The max value is {max_val} on position {k_val}\n")

        if max_val == 0.0:
            max_val = np.where(np.abs(restartable_samples) == max_residu)[0][0]
            k_val = restartable_indices[max_val][0]
            print(f"The max value is now {max_val} on position {k_val}\n")

        t_deltas.append(max_val)
        Ks.append(k_val)

        # Reevaluate the samples
        for p in range(sample_size):
            k = sample_indices[p, 0]
            j = sample_indices[p, 1]
            i = sample_indices[p, 2]
            sample_values[p] = sample_values[p] - ((1.0 / r_deltas[rank-1]) * (1.0 / t_deltas[rank-1]) * cols[rank-1][i] * rows[rank-1][j] * tubes[rank-1][k])

        # Find the maximum error on the samples
        if restartable_samples.size == 0:
            max_residu = 0
        else:
            max_residu = np.max(np.abs(restartable_samples))

        # Update stopping criteria
        remaining_average = np.average(np.square(sample_values))
        max_allowed_relative_error = 0  # TODO
        stopcrit = (sqrt(remaining_average) < max_allowed_relative_error)

        factors_aca = aca_as_cp(cols, rows, tubes, c_deltas, r_deltas)
        aca_norm = compare_cp_with_full(cp=factors_aca, original=tensor)
        aca_vects_norms.append(aca_norm)

        rank += 1

    return aca_vects_norms


def find_largest_absolute_value(fiber):
    max_abs_value = None
    max_value = None
    index = None

    for i, num in enumerate(fiber):
        abs_value = abs(num)
        if max_abs_value is None or abs_value > max_abs_value:
            max_abs_value = abs_value
            max_value = num
            index = i

    if max_value is None:
        return None, None

    return max_value, index


def set_to_zero(indices, numbers):
    for index in indices:
        if index < len(numbers):
            numbers[index] = 0
    return numbers


def get_fiber(i=None, j=None, k=None):
    fiber = []

    if i is None:
        fiber = TENSOR[k, j, :]
    elif j is None:
        fiber = TENSOR[k, :, i]
    elif k is None:
        fiber = TENSOR[:, j, i]

    return fiber


def get_element(i, j, k):
    return TENSOR[k, j, i]


def generate_samples(tensor, amount):
    shape = tensor.shape
    sample_indices = np.zeros(shape=(amount, 3), dtype=int)
    sample_values = np.zeros(amount, dtype=float)

    for a in range(amount):
        i = random.randint(0, shape[1]-1)
        j = random.randint(0, shape[2]-1)
        while i == j:
            j = random.randint(0, shape[2]-1)
        k = random.randint(0, shape[0]-1)
        sample_indices[a, 0] = k
        sample_indices[a, 1] = j
        sample_indices[a, 2] = i
        sample_values[a] = get_element(i, j, k)

    return sample_indices, sample_values


def compare_aca_original(matrices, tubes, m_delta, original):
    t = np.zeros(original.shape)
    # Make whole tensor as sum of product of matrices and tube vectors
    # print("ts", tubes)
    for i in range(len(matrices)):
        matrix = np.divide(matrices[i], m_delta[i])
        # print("m", matrix)
        tube = tubes[i]
        # print("t", tube)
        t += np.einsum('i,jk->ijk', tube, matrix)

    # Fill diagonal of each frontal matrix in tensor with zeros
    for i in range(len(t)):
        np.fill_diagonal(t[i], 0)
    # print("einsum \n", t)

    # Compare with original tensor
    difference = original-t
    f_norm = np.linalg.norm(difference)
    original_norm = np.linalg.norm(original)
    norm = f_norm/original_norm
    return norm


def aca_as_cp(cols, rows, tubes, row_delta, tube_delta):
    """
    Converts the result of ACA to the CP-format to easily reconstruct the full tensor.
    :param cols: the column vectors from ACA
    :param rows: the rows vectors from ACA
    :param tubes: the tubes vectors from ACA
    :param row_delta: the deltas from ACA, used to calculate weights
    :param tube_delta: the deltas from ACA, used to calculate weights
    :return: returns (weights, factors) based on ACA result
    """
    f_cs = np.array(np.transpose(cols))
    f_rs = np.array(np.transpose(rows))
    f_ts = np.array(np.transpose(tubes))
    combined_fs = [f_ts, f_cs, f_rs]
    weights = np.ones(len(row_delta))
    for w in range(len(weights)):
        print("row_D", row_delta[w], "tube_D", tube_delta[w])
        weights[w] = (1/row_delta[w]) * (1/tube_delta[w])
    return weights, combined_fs


def compare_cp_with_full(cp, original):
    reconstructed = tl.cp_tensor.cp_to_tensor(cp)

    # Make all elements on diagonal 0.
    for i in range(len(reconstructed)):
        np.fill_diagonal(reconstructed[i], 0)
    print("R", reconstructed)

    difference = original-reconstructed
    # Calculate tensor norm (~Frobenius matrix norm)
    f_norm = np.linalg.norm(difference)
    original_norm = np.linalg.norm(original)
    norm = f_norm/original_norm
    return norm


def random_tensor(shape, low, high, seed):
    np.random.seed(seed)

    tensor = np.zeros(shape)
    for i in range(shape[0]):
        # Generate a random symmetric 3-by-3 matrix
        sym_matrix = np.random.randint(low, high, size=(shape[1], shape[1]))
        sym_matrix = (sym_matrix + sym_matrix.T) // 2
        np.fill_diagonal(sym_matrix, 0)

        tensor[i, :, :] = sym_matrix

    return tensor


path = "tensors/person2all_ex_75ts.npy"
# TENSOR = np.load(path)
TENSOR = random_tensor((3, 3, 3), 1, 10, seed=1)
print(TENSOR)
maxrank = 3

res = aca_tensor(tensor=TENSOR, max_rank=maxrank+1, random_seed=0)
print(res)
plotting.plot_norms_aca_cp(maxrank, aca_v=res, aca_k=None, aca_m=None, cp_norms=None)

