import random
import numpy as np
import tensorly as tl

# Set print to print matrices and vectors in one line
np.set_printoptions(linewidth=np.inf)


def aca_tensor(tensor, max_rank, start_col=None, random_seed=None, to_cluster=False):
    rows = []
    cols = []
    tubes = []
    r_deltas = []
    c_deltas = []
    t_deltas = []

    x_used = []
    y_used = []
    z_used = []
    reconstructed = None

    aca_vects_norms = []

    # Def shape of tensor
    shape = tensor.shape  # shape = (z, y, x)
    print(shape)

    # If a random seed is given, use this.
    if random_seed is not None:
        random.seed(random_seed)

    # amount = 2
    # sample_indices = np.zeros(shape=(amount, 3), dtype=int)
    # sample_values = np.zeros(amount, dtype=float)
    # a = 0
    # sample_indices[a, 0] = 1
    # sample_indices[a, 1] = 0
    # sample_indices[a, 2] = 1
    # sample_values[a] = get_element(1, 0, 1, tensor)
    # a = 1
    # sample_indices[a, 0] = 0
    # sample_indices[a, 1] = 1
    # sample_indices[a, 2] = 0
    # sample_values[a] = get_element(0, 1, 0, tensor)

    # Generate random samples for stopping criteria
    sample_indices, sample_values = generate_samples(tensor, max_rank)
    print(f"Sample indices: {sample_indices} \n with sample values {sample_values}")
    sample_size = len(sample_values)

    # If no start column is given, initialize a random one.
    if start_col is None:
        max_sample = np.max(np.abs(sample_values))
        index_sample_max = np.where(np.abs(sample_values) == max_sample)[0][0]

        x_as = sample_indices[index_sample_max, 1]
        z_as = sample_indices[index_sample_max, 0]
        print(f"chosen col: x={x_as}, z={z_as}")

    else:
        x_as = start_col[0]
        z_as = start_col[1]

    x_used.append(x_as)
    z_used.append(z_as)

    rank = 0
    # Select new skeletons until desired rank is reached.
    while rank < max_rank:

        print(f"--------------------- RANK {rank} ----------------------")
        # print(tensor[z_as])

        # --------- COLUMNS ---------
        col_fiber = get_fiber(tensor, k=z_as, i=x_as)
        print("col before", col_fiber)

        if reconstructed is None:
            approx = np.zeros(len(col_fiber))
            for i in range(rank):
                approx = np.add(approx, cols[i] * rows[i][x_as] * tubes[i][z_as] * (1.0 / c_deltas[i]) * (1.0 / r_deltas[i]))
            print("A:", approx)
        else:
            approx = get_fiber(reconstructed, k=z_as, i=x_as)
            print("A:", approx)

        new_col = np.subtract(col_fiber, approx)
        print("col after", new_col)

        previous = [i for i, item in enumerate(z_used[0:len(z_used)-1]) if item == z_as]
        print("previous y", y_used)
        to_delete = [y_used[p] for p in previous]
        print("y to delete", to_delete)
        col_without_previous = set_to_zero(to_delete, new_col)
        max_val, y_as = find_largest_absolute_value(col_without_previous)

        print(f"max val: {max_val} on Y pos: {y_as}")
        cols.append(new_col)
        c_deltas.append(max_val)
        y_used.append(y_as)

        # --------- ROWS ---------
        row_fiber = get_fiber(tensor, k=z_as, j=y_as)
        print("r before", row_fiber)

        if reconstructed is None:
            approx = np.zeros(len(row_fiber))
            for i in range(rank):
                approx = np.add(approx,
                                cols[i][y_as] * rows[i] * tubes[i][z_as] * (1.0 / c_deltas[i]) * (1.0 / r_deltas[i]))
            print("A", approx)
        else:
            approx = get_fiber(reconstructed, k=z_as, j=y_as)
            print("A:", approx)

        new_row = np.subtract(row_fiber, approx)
        print("r after", new_row)

        new_abs_row = abs(new_row)
        previous = [i for i, item in enumerate(y_used[0:len(y_used)-1]) if item == y_as]
        # print("prevous x", x_used)

        # Needed when adding first chosen column
        to_delete = [x_used[p+1] for p in previous]

        # to_delete = [x_used[p] for p in previous]
        # print("x to delete", to_delete)
        row_without_previous = set_to_zero(to_delete, new_row)
        max_val, x_as = find_largest_absolute_value(row_without_previous)

        print(f"max val: {max_val} on X pos: {x_as}")
        rows.append(new_row)
        r_deltas.append(max_val)
        x_used.append(x_as)

        # --------- TUBES ---------
        print(x_used)
        print("takes:", y_as, ", ", x_as)
        tube_fiber = get_fiber(tensor, j=y_as, i=x_as)
        print("t:", tube_fiber)

        if reconstructed is None:
            approx = np.zeros(len(tube_fiber))
            for i in range(rank):
                approx = np.add(approx,
                                cols[i][y_as] * rows[i][x_as] * tubes[i] * (1.0 / c_deltas[i]) * (1.0 / r_deltas[i]))
        else:
            approx = get_fiber(reconstructed, j=y_as, i=x_as)
            print("A:", approx)

        new_tube = np.subtract(tube_fiber, approx)
        print("t after:", new_tube)

        previous = [i for i, item in enumerate(x_used[0:len(x_used)-1]) if item == x_as]
        print("previous z", z_used)
        to_delete = [z_used[p] for p in previous]
        print("z to delete", to_delete)
        t = new_tube.copy()
        tube_without_previous = set_to_zero(to_delete, t)
        z_max, z_as = find_largest_absolute_value(tube_without_previous)

        print(f"max val: {z_max} on Z pos: {z_as}")

        tubes.append(new_tube)
        t_deltas.append(z_max)
        z_used.append(z_as)

        factors_aca = aca_as_cp(cols, rows, tubes, c_deltas, r_deltas)
        reconstructed = reconstruct_tensor(factors_aca, symm=True)
        print(reconstructed)
        print("diff\n", tensor-reconstructed)
        aca_norm = compare_cp_with_full(cp=factors_aca, original=tensor)
        aca_vects_norms.append(aca_norm)

        rank += 1

    if to_cluster:
        return cols, rows, tubes, c_deltas, r_deltas, r_deltas
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


def set_to_zero(indices, numbers):
    for index in indices:
        if index < len(numbers):
            numbers[index] = 0
    return numbers


def aca_k_vectors(tensor, max_rank, k_hat, start_tube=None, random_seed=None):
    print("shape", tensor.shape)

    rows = []
    cols = []
    tubes = []
    r_deltas = []
    c_deltas = []
    t_deltas = []

    x_used = []
    y_used = []
    z_used = []

    matrices = []
    matrix_idx = []
    m_deltas = []
    aca_k_hat_norms = []

    # Def shape of tensor
    shape = tensor.shape  # shape = (z, y, x)

    # If a random seed is given, use this.
    if random_seed is not None:
        random.seed(random_seed)

    # If no start column is given, initialize a random one.
    if start_tube is None:
        x_as = random.randint(0, shape[2]-1)
        y_as = random.randint(0, shape[1]-1)
        while y_as == x_as:
            y_as = random.randint(0, shape[1]-1)
        print(f"chosen tube: x={x_as}, y={y_as}")

    else:
        x_as = start_tube[0]
        y_as = start_tube[1]

    x_used.append(x_as)
    y_used.append(y_as)

    rank = 0

    # Select new skeletons until desired rank is reached.
    while rank < max_rank:

        print(f"--------------------- RANK {rank} ----------------------")
        print("x_used", x_used)
        print("y_used", y_used)
        print("z_used", z_used)
        # print(tensor[z_as])
        #  --------- TUBES ---------
        tube_fiber = tensor[:, y_as, x_as]
        print("t:", tube_fiber)

        approx = np.zeros(len(tube_fiber))
        for i in range(rank):
            approx = np.add(approx, cols[i][y_as] * rows[i][x_as] * tubes[i] * (1.0 / c_deltas[i]) * (1.0 / r_deltas[i]))
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

        # Save cols and rows in subset
        k_cols = []
        c_ds = []
        yc_used = y_used.copy()
        k_rows = []
        r_ds = []
        xr_used = x_used.copy()
        # Loop to select k_hat cols and rows for current tube
        k = 0
        m_tot = np.zeros(shape=(shape[1], shape[2]))
        while k < k_hat:
            # --------- COLUMNS ---------
            col_fiber = tensor[z_as, :, x_as]
            print("col before", col_fiber)

            approx = np.zeros(len(col_fiber))
            for i in range(k):
                approx = np.add(approx, k_cols[i] * k_rows[i][x_as] * (1.0 / r_ds[i]) )
            new_col = np.subtract(col_fiber, approx)
            print("col after", new_col)

            new_abs_col = abs(new_col)
            col_without_previous = np.delete(new_abs_col, yc_used, axis=0)
            max_val_y = np.max(col_without_previous)
            y_as = np.where(new_abs_col == max_val_y)[0][0]

            print(f"max val: {max_val_y} on Y pos: {y_as}")

            k_cols.append(new_col)
            c_ds.append(max_val_y)
            yc_used.append(y_as)

            # --------- ROWS ---------
            row_fiber = tensor[z_as, y_as, :]
            print("r before", row_fiber)

            approx = np.zeros(len(row_fiber))
            for i in range(k):
                approx = np.add(approx, k_cols[i][y_as] * k_rows[i] * (1.0 / c_ds[i]) )
            new_row = np.subtract(row_fiber, approx)
            print("r after", new_row)

            new_abs_row = abs(new_row)
            row_without_previous = np.delete(new_abs_row, xr_used, axis=0)
            max_val_x = np.max(row_without_previous)
            x_as = np.where(new_abs_row == max_val_x)[0][0]

            print(f"max val: {max_val_x} on X pos: {x_as}")

            k_rows.append(new_row)
            r_ds.append(max_val_x)
            xr_used.append(x_as)

            k += 1

            new_row = np.divide(new_row, max_val_x)
            matrix = np.outer(new_col, new_row)
            m_tot += matrix
            print("matrix", matrix.shape)

        matrices.append(m_tot)
        m_idx = np.unravel_index(np.argmax(abs(m_tot)), m_tot.shape)
        max_val = m_tot[m_idx]
        m_deltas.append(max_val)

        x_used.append(x_as)
        y_used.append(y_as)

        aca_norm = compare_aca_original(matrices, tubes, m_deltas, tensor)
        aca_k_hat_norms.append(aca_norm)

        cols.append(new_col)
        c_deltas.append(max_val_y)
        rows.append(new_row)
        r_deltas.append(max_val_x)
        rank += 1
    return aca_k_hat_norms


def aca_matrix_x_vector(tensor, max_rank, start_matrix=None, random_seed=None, to_cluster=False):
    # The already chosen matrices and tubes are stored.
    matrices = []
    tubes = []
    # The found deltas are stored.
    m_deltas = []
    t_deltas = []
    # The index of matrices and tubes are stored to make sure they are not used twice.
    matrix_idx = []
    tubes_idx = []

    aca_matrix_norms = []

    # Def shape of tensor
    shape = tensor.shape  # shape = (z, y, x)

    # If a random seed is given, use this.
    if random_seed is not None:
        random.seed(random_seed)

    # Generate samples for stopping criteria and better start
    sample_indices, sample_values = generate_tube_samples(tensor)
    print(f"Sample indices: {sample_indices} \n with sample values {sample_values}")
    sample_size = len(sample_values)

    # If no start column is given, initialize a random one.
    if start_matrix is None:
        max_sample = np.max(np.abs(sample_values))
        index_sample_max = np.where(np.abs(sample_values) == max_sample)[0][0]

        z_as = sample_indices[index_sample_max, 0]
        print(f"chosen matrix: z={z_as}")

    else:
        z_as = start_matrix
    tubes_idx.append(z_as)

    rank = 0
    # Select new skeletons until desired rank is reached.
    while rank < max_rank:

        print(f"--------------------- RANK {rank} ----------------------")
        # print(tensor[z_as])

        # --------- MATRICES ---------
        matrix = tensor[z_as, :, :]
        print("matrix before", z_as, "; ", matrix)

        approx = np.zeros(matrix.shape)
        for i in range(rank):
            approx = np.add(approx, matrices[i] * tubes[i][z_as] * (1.0 / m_deltas[i]))
            # print("approx: \n", approx)
        new_matrix = np.subtract(matrix, approx)
        # print("matrix after: \n", new_matrix)

        new_abs_matrix = abs(new_matrix)
        # Setting previously used scores to zero to make sure they are not used twice
        for idx in matrix_idx:
            # symmetric matrix
            i, j = idx
            new_abs_matrix[i][j] = 0
            new_abs_matrix[j][i] = 0

        m_idx = np.unravel_index(np.argmax(new_abs_matrix), new_matrix.shape)
        max_val = new_matrix[m_idx]
        print(f"max val: {max_val} on Y pos: {m_idx}")

        matrices.append(new_matrix)
        m_deltas.append(max_val)
        matrix_idx.append(m_idx)

        # --------- TUBES ---------
        y_as = m_idx[0]
        x_as = m_idx[1]
        tube_fiber = tensor[:, y_as, x_as]
        print("t:", m_idx, "; ", tube_fiber)

        approx = np.zeros(len(tube_fiber))
        for i in range(rank):
            approx = np.add(approx, matrices[i][m_idx] * tubes[i] * (1.0 / m_deltas[i]))
        new_tube = np.subtract(tube_fiber, approx)

        print("t after:", new_tube)
        new_abs_tube = abs(new_tube)
        tube_without_previous = np.delete(new_abs_tube, tubes_idx, axis=0)
        max_val = np.max(tube_without_previous)
        z_as = np.where(new_abs_tube == max_val)[0][0]
        print(f"max val: {max_val} on Z pos: {z_as}")

        tubes.append(new_tube)
        t_deltas.append(max_val)
        tubes_idx.append(z_as)

        aca_norm = compare_aca_original(matrices, tubes, m_deltas, tensor)
        aca_matrix_norms.append(aca_norm)

        rank += 1

    if to_cluster:
        return matrices, tubes, m_deltas, t_deltas
    else:
        return aca_matrix_norms


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
    print(t)
    # Fill diagonal of each frontal matrix in tensor with zeros
    for i in range(len(t)):
        np.fill_diagonal(t[i], 0)

    # Compare with original tensor
    difference = original-t
    print("DIFFERENCE \n", difference)
    f_norm = np.linalg.norm(difference)
    original_norm = np.linalg.norm(original)
    original_norm2 = calc_norm(difference)
    print("check =", original_norm2)
    print("original norm = ", original_norm)
    print("fnorm =", f_norm)
    norm = f_norm/original_norm
    return norm


def aca_as_cp(cols, rows, tubes, col_delta, row_delta):
    """
    Converts the result of ACA to the CP-format to easily reconstruct the full tensor.
    :param cols: the column vectors from ACA
    :param rows: the rows vectors from ACA
    :param tubes: the tubes vectors from ACA
    :param col_delta: the deltas from ACA, used to calculate weights
    :param row_delta: the deltas from ACA, used to calculate weights
    :return: returns (weights, factors) based on ACA result
    """
    f_cs = np.array(np.transpose(cols))
    f_rs = np.array(np.transpose(rows))
    f_ts = np.array(np.transpose(tubes))
    combined_fs = [f_ts, f_cs, f_rs]
    weights = np.ones(len(row_delta))
    for w in range(len(weights)):
        # print("col_D", col_delta[w], "row_D", row_delta[w])
        weights[w] = (1/col_delta[w]) * (1/row_delta[w])
    return weights, combined_fs


def reconstruct_tensor(cp, symm=True):
    reconstructed = tl.cp_tensor.cp_to_tensor(cp)

    # Make all elements on diagonal 0.
    for i in range(len(reconstructed)):
        np.fill_diagonal(reconstructed[i], 0)
        if symm:
            rit = reconstructed[i].T
            reconstructed[i] = reconstructed[i]+rit
    return reconstructed


def compare_cp_with_full(cp, original):
    reconstructed = np.abs(reconstruct_tensor(cp, symm=True))
    difference = np.subtract(original, reconstructed)

    # Calculate tensor norm (~Frobenius matrix norm)
    # f_norm2 = calc_norm(difference)
    f_norm = np.linalg.norm(difference)
    original_norm = np.linalg.norm(original)
    # print("original norm = ", original_norm)
    # print("fnorm =", f_norm)
    original_norm2 = calc_norm(difference)
    # print("check =", original_norm2)
    norm = f_norm/original_norm
    # norm2 = f_norm2/original_norm2
    return norm


def calc_norm(tensor):
    s = 0
    shape = tensor.shape
    for k in range(shape[0]):
        for j in range(shape[1]):
            for i in range(shape[2]):
                temp = tensor[k, j, i]
                s += (temp * temp)
    res = np.sqrt(s)
    return res


def generate_tube_samples(tensor):
    zs, ys, xs = tensor.shape
    sample_indices = np.zeros(shape=(zs, 3), dtype=int)
    sample_values = np.zeros(zs, dtype=float)

    # Cycle over all horizontal slices, then take random element of that slice
    for z in range(zs):
        x = random.randint(0, xs-1)
        y = random.randint(0, ys-1)
        while x == y:
            y = random.randint(0, ys - 1)

        sample_indices[z, 0] = z
        sample_indices[z, 1] = y
        sample_indices[z, 2] = x
        sample_values[z] = get_element(x, y, z, tensor)

    return sample_indices, sample_values


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
        sample_values[a] = get_element(i, j, k, tensor)

    return sample_indices, sample_values


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
