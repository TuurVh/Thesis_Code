import random
import numpy as np
import random as rnd
from tensorly.decomposition import parafac

# Set print to print matrices and vectors in one line
np.set_printoptions(linewidth=np.inf)


def generate_samples_less(tensor):
    k, j, i = tensor.shape
    inds = np.zeros(shape=(k, 3), dtype=int)
    vals = np.zeros(k, dtype=float)

    for mat in range(0, k):
        z = mat
        x = rnd.randint(0, i - 1)
        y = rnd.randint(0, j - 1)
        while x == y:
            y = rnd.randint(0, j - 1)
        inds[mat, 0] = z
        inds[mat, 1] = x
        inds[mat, 2] = y
        print(x, y, z)
        vals[mat] = get_element(x, y, z, tensor)
    return inds, vals


def generate_samples(tensor):
    k, j, i = tensor.shape
    all_sample_indices = None
    all_sample_values = None
    amount = 0
    max_amount = 1
    while amount < max_amount:
        sample_indices = None
        sample_values = None
        for mat in range(0, k):
            inds = np.zeros(shape=(j, 3), dtype=int)
            vals = np.zeros(j, dtype=float)
            for row in range(0, j):
                z = mat
                y = row
                x = random.randint(0, i - 1)
                while x == y:
                    x = random.randint(0, i - 1)
                inds[row, 0] = z
                inds[row, 1] = y
                inds[row, 2] = x
                vals[row] = get_element(x, y, z, tensor)
            if sample_indices is None:
                sample_indices = inds
                sample_values = vals
            else:
                sample_indices = np.concatenate((sample_indices, inds))
                sample_values = np.concatenate((sample_values, vals))
        if all_sample_indices is None:
            all_sample_indices = sample_indices
            all_sample_values = sample_values
        else:
            all_sample_indices = np.concatenate((all_sample_indices, sample_indices))
            all_sample_values = np.concatenate((all_sample_values, sample_values))
        amount += 1

    return all_sample_indices, all_sample_values


def get_element(i, j, k, tensor):
    return tensor[k, j, i]


def get_fiber(tensor, i=None, j=None, k=None):
    fiber = []

    if i is None:
        fiber = tensor[k, j, :]
    elif j is None:
        fiber = tensor[k, :, i]
    elif k is None:
        fiber = tensor[:, j, i]

    return fiber


def set_to_zero(indices, numbers):
    for index in indices:
        if index < len(numbers):
            numbers[index] = 0
    return numbers


def get_CP_decomposition(tensor, max_rank):
    factors = parafac(tensor, rank=max_rank, normalize_factors=False)[1]
    return factors


def aca_tensor(tensor, max_rank, random_seed=None, to_cluster=False):
    rows = []
    cols = []
    tubes = []

    r_deltas = []
    c_deltas = []

    x_used = []
    y_used = []
    z_used = []

    aca_vects_norms = []

    # Shape of given tensor
    shape = tensor.shape  # shape = (z, y, x)

    # If a random seed is given, use this
    if random_seed is not None:
        random.seed(random_seed)

    # Generate random samples for stopping criteria
    sample_indices, sample_values = generate_samples(tensor)
    print(f"Sample indices: {sample_indices} \n with sample values {sample_values}")
    sample_size = len(sample_values)
    print("amount samples", sample_size)

    # Initialize starting row based on the samples
    index_sample_max = np.argmax(np.abs(sample_values))
    max_sample = sample_values[index_sample_max]

    y_as = sample_indices[index_sample_max, 2]
    z_as = sample_indices[index_sample_max, 0]

    rank = 0

    # Select new skeletons until desired rank is reached.
    while rank < max_rank:
        print(" -- RANK ", rank+1, " --")
        # works with symmetric frontal matrices so take the same index row and column!
        # --------- ROWS ---------
        row_fiber = get_fiber(tensor, k=z_as, j=y_as)
        print("row before", row_fiber)
        approx = np.zeros(len(row_fiber))
        for r in range(rank):
            approx = np.add(approx, rows[r] * cols[r][y_as] * tubes[r][z_as] * (1.0/r_deltas[r]) * (1.0/c_deltas[r]))

        new_row = np.subtract(row_fiber, approx)
        print("newrw", new_row)

        # Update row to not choose same again

        r_max_val, x_as = find_largest_absolute_value(new_row)
        print("max", r_max_val)

        if r_max_val == 0.0:
            index_sample_max = np.argmax(np.abs(sample_values))
            [z_as, x_as, y_as] = sample_indices[index_sample_max]
            print("takes", sample_indices[index_sample_max] )
            continue
        print("x_as", x_as)

        # --------- COLS ---------
        col_fiber = get_fiber(tensor, k=z_as, i=x_as)
        print("col before", col_fiber)
        approx = np.zeros(len(col_fiber))
        for r in range(rank):
            approx = np.add(approx, rows[r][x_as] * cols[r] * tubes[r][z_as] * (1.0/r_deltas[r]) * (1.0/c_deltas[r]))
        print("approx", approx)
        new_col = np.subtract(col_fiber, approx)
        print("newcol", new_col)

        c_max_val, y_as = find_largest_absolute_value(new_col)
        print("y_as", y_as)

        # --------- TUBES ---------
        tube_fiber = get_fiber(tensor, i=x_as, j=y_as)
        print("tubef", tube_fiber)
        approx = np.zeros(len(tube_fiber))
        for r in range(rank):
            approx = np.add(approx, rows[r][x_as] * cols[r][y_as] * tubes[r] * (1.0/r_deltas[r]) * (1.0/c_deltas[r]))
        print("approx", approx)

        new_tube = np.subtract(tube_fiber, approx)
        print("NT", new_tube)

        # Update tube to not choose same row again
        if rank > 0:
            if y_used[rank-1] == y_as:
                to_delete = z_used[rank-1]
                tube_without_previous = new_tube.copy()
                tube_without_previous[to_delete] = 0
            else:
                tube_without_previous = new_tube
        else:
            tube_without_previous = new_tube

        t_max_val, z_as = find_largest_absolute_value(tube_without_previous)
        print('z_as', z_as)

        # Append all
        r_deltas.append(r_max_val)
        c_deltas.append(c_max_val)
        rows.append(new_row)
        cols.append(new_col)
        tubes.append(new_tube)

        x_used.append(x_as)
        y_used.append(y_as)
        z_used.append(z_as)

        # ----- REEVALUATE SAMPLES -----
        for s in range(sample_size):
            x_ = sample_indices[s, 1]
            y_ = sample_indices[s, 2]
            z_ = sample_indices[s, 0]
            temp_sample = sample_values[s]
            sample_values[s] = temp_sample - (rows[rank][x_] * cols[rank][y_] * tubes[rank][z_] *
                                              (1.0/c_deltas[rank]) * (1.0 / r_deltas[rank]))
        print("new vals", sample_values)
        # Find new largest value in sample_values
        index_sample_max = np.argmax(np.abs(sample_values))
        max_sample = sample_values[index_sample_max]

        print("z", t_max_val, "max res", max_sample)
        if np.abs(t_max_val) < max_sample - 0.001:
            print("so on pos:", sample_indices[index_sample_max])
            [z_as, x_as, y_as] = sample_indices[index_sample_max]

        matrices = preprocess_to_matrices(cols, rows, r_deltas)
        aca_norm = compare_aca_original(matrices, tubes, c_deltas, tensor)
        aca_vects_norms.append(aca_norm)

        rank += 1

    if to_cluster:
        return rows, cols, tubes, r_deltas, c_deltas
    else:
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


def preprocess_to_matrices(cols, rows, r_deltas):
    matrices = []
    for idx, col in enumerate(cols):
        matrix = np.outer(rows[idx], col) / r_deltas[idx]
        matrices.append(matrix)
    return matrices


def compare_aca_original(matrices, tubes, m_delta, original):
    t = np.zeros(original.shape)
    # Make whole tensor as sum of product of matrices and tube vectors
    for i in range(len(matrices)):
        matrix = np.divide(matrices[i], m_delta[i])
        tube = tubes[i]
        t += np.einsum('i,jk->ijk', tube, matrix)
    print("RECON", t)
    # Fill diagonal of each frontal matrix in tensor with zeros
    for i in range(len(t)):
        np.fill_diagonal(t[i], 0)

    # Compare with original tensor
    difference = original-t
    print("DIFFERENCE \n", difference)
    f_norm = np.linalg.norm(difference)
    original_norm = np.linalg.norm(original)
    norm = f_norm/original_norm
    return norm
