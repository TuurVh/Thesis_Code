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

    used_tubes = []
    x_used = []
    z_used = []

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
    sample_indices, sample_values = generate_samples(tensor)
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
    # x_used.append(x_as)
    # z_used.append(z_as)

    restartable_samples = sample_values
    restartable_indices = sample_indices
    deleted_indices = np.array([], dtype=int)

    rank = 0
    # Select new skeletons until desired rank is reached.
    while rank < max_rank:

        print(f"--------------------- RANK {rank} ----------------------")
        # print(tensor[z_as])

        # works with symmetric frontal matrices so take the same index row and column!
        # --------- COLUMNS ---------
        col_fiber = get_fiber(tensor, k=z_as, i=x_as)
        print("col before", col_fiber)

        approx = np.zeros(len(col_fiber))
        for i in range(rank):
            approx = np.add(approx, cols[i] * cols[i][x_as] * tubes[i][z_as] * (1.0 / c_deltas[i]) * (1.0 / r_deltas[i]))
        print("A:", approx)

        new_col = np.subtract(col_fiber, approx)
        print("col after", new_col)

        # previous = [i for i, item in enumerate(z_used[0:len(z_used)-1]) if item == z_as]
        # print("previous y", x_used)
        # to_delete = [x_used[p] for p in previous]
        # print("y to delete", to_delete)
        # col_without_previous = set_to_zero(to_delete, new_col.copy())

        col_with_zero = set_to_zero([x_as], new_col.copy())
        print("col w/ 0:", col_with_zero)

        temp_d = new_col[x_as]
        if temp_d == 0.0:
            print("delta == 0")
            new_d = np.max(np.abs(new_col))
            # If col is perfectly approximated
            if new_d == 0.0:
                x_as = restartable_indices[max_residu_idx, 2]
                y_as = restartable_indices[max_residu_idx, 1]
                z_as = restartable_indices[max_residu_idx, 0]
                # x_used.append(x_as)
                # z_used.append(z_as)
                continue
            temp_d = new_d

        max_val, y_as = find_largest_absolute_value(col_with_zero)

        print(f"max val: {max_val} on Y pos: {y_as}")

        cols.append(new_col)
        c_deltas.append(max_val)
        x_used.append(x_as)

        r_deltas.append(temp_d)

        # print("outer", np.outer(new_col, new_col)/temp_d)
        # print("outer zonder", np.outer(new_col, new_col))
        used_tubes.append((x_as, y_as))

        # --------- TUBES ---------
        print("takes:", y_as, ", ", x_as)
        tube_fiber = get_fiber(tensor, j=y_as, i=x_as)
        print("t:", tube_fiber)

        approx = np.zeros(len(tube_fiber))
        for i in range(rank):
            approx = np.add(approx, cols[i][y_as] * cols[i][x_as] * tubes[i] * (1.0 / c_deltas[i]) * (1.0 / r_deltas[i]))

        new_tube = np.subtract(tube_fiber, approx)
        print("t after:", new_tube)

        # for prev in z_used:
        #     print("prev", prev, "z", z_as)
        #     if prev != z_as:
        #         new_tube[prev] = 0
        # print("NT", new_tube)

        previous = [i for i, item in enumerate(x_used[0:len(x_used)-1]) if item == x_as]
        print("previous z", z_used)
        to_delete = [z_used[p] for p in previous]
        print("z to delete", to_delete)
        t = new_tube.copy()
        tube_without_previous = set_to_zero(to_delete, t)

        z_max, z_as = find_largest_absolute_value(new_tube)

        print(f"max val: {z_max} on Z pos: {z_as}")

        tubes.append(new_tube)
        t_deltas.append(z_max)
        z_used.append(z_as)

        # ----- REEVALUATE SAMPLES -----
        for s in range(sample_size):
            x_ = sample_indices[s, 1]
            y_ = sample_indices[s, 2]
            z_ = sample_indices[s, 0]
            temp_sample = sample_values[s]
            sample_values[s] = temp_sample - (cols[rank][x_] * cols[rank][y_] * tubes[rank][z_] *
                                              (1.0/c_deltas[rank]) * (1.0 / r_deltas[rank]))

        # --- DELETE Samples ---
        # print('looking for', z_as, y_as, x_as)
        # print('idx', sample_indices)
        # print('vals', sample_values)
        idx_on_row = np.where(((sample_indices[:, 1] == y_as) & (sample_indices[:, 0] == z_as)))[0]
        idx_on_tube = np.where(((sample_indices[:, 1] == y_as) & (sample_indices[:, 2] == x_as)))[0]
        print(idx_on_row, idx_on_tube, "-----------------------")
        idx_in_samples = np.concatenate((idx_on_row, idx_on_tube))
        if deleted_indices.size == 0:
            deleted_indices = idx_in_samples
        else:
            deleted_indices = np.concatenate((deleted_indices, idx_in_samples), axis=0)
        restartable_samples = np.delete(sample_values, deleted_indices, axis=0)
        restartable_indices = np.delete(sample_indices, deleted_indices, axis=0)
        print("left over", restartable_indices)

        # Find the maximum error on the samples
        if restartable_samples.size == 0:
            max_residu = 0
        else:
            max_residu_idx = np.argmax(np.abs(restartable_samples))
            max_residu = restartable_samples[max_residu_idx]

        # If max value of tube is smaller than max value of samples, use max sample as next starting point
        print("resterende samples", restartable_samples)
        print("Z:", z_max, ", max res:", max_residu)
        if abs(z_max) < max_residu - 0.001:
            x_as = restartable_indices[max_residu_idx, 1]
            y_as = restartable_indices[max_residu_idx, 2]
            z_as = restartable_indices[max_residu_idx, 0]
            x_used.append(x_as)
            z_used.append(z_as)

        # --------- RECONSTRUCTION ---------
        # factors_aca = aca_as_cp(cols, cols, tubes, r_deltas, c_deltas)
        # reconstructed = reconstruct_tensor(factors_aca, zeros=False)
        # print("--- reconstr:", reconstructed)
        # print("diff\n", tensor-reconstructed)
        # aca_norm = compare_cp_with_full(cp=factors_aca, original=tensor)

        matrices = preprocess_to_matrices(cols, r_deltas, used_tubes)
        aca_norm = compare_aca_original(matrices, tubes, c_deltas, tensor)
        aca_vects_norms.append(aca_norm)

        x_as = y_as
        rank += 1

    if to_cluster:
        return cols, rows, tubes, c_deltas, r_deltas, r_deltas
    else:
        return aca_vects_norms


def preprocess_to_matrices(cols, r_deltas, used_tubes):
    # used_tubes = used_tubes[1:]
    # print("used tubes:", used_tubes)
    matrices = []
    for idx, col in enumerate(cols):
        matrix = np.outer(col, col) / r_deltas[idx]
        print("idx", idx, 'col', col, "r_d", r_deltas[idx])
        print("outer:", matrix)
        # if idx > 0:
        #     for i in range(idx-1, 0, -1):
        #         x, y = used_tubes[i]
        #         matrix[x][y] = 0
        #         matrix[y][x] = 0
        #     print("updated matrix", matrix)
        matrices.append(matrix)
    return matrices


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

        # from vector method:
        # works with symmetric frontal matrices so take the same index row and column!
        # --------- COLUMNS ---------
        col_fiber = get_fiber(tensor, k=z_as, i=x_as)
        print("col before", col_fiber)

        approx = np.zeros(len(col_fiber))
        for i in range(rank):
            approx = np.add(approx, cols[i] * cols[i][x_as] * tubes[i][z_as] * (1.0 / c_deltas[i]) * (1.0 / r_deltas[i]))
        print("A:", approx)

        new_col = np.subtract(col_fiber, approx)
        print("col after", new_col)

        # previous = [i for i, item in enumerate(z_used[0:len(z_used)-1]) if item == z_as]
        # print("previous y", y_used)
        # to_delete = [y_used[p] for p in previous]
        # print("y to delete", to_delete)
        # col_without_previous = set_to_zero(to_delete, new_col.copy())

        col_with_zero = set_to_zero([x_as], new_col.copy())
        print("col w/ 0:", col_with_zero)
        max_val, y_as = find_largest_absolute_value(col_with_zero)

        print(f"max val: {max_val} on Y pos: {y_as}")
        cols.append(new_col)
        c_deltas.append(max_val)
        x_used.append(x_as)

        temp_d = new_col[x_as]
        if temp_d == 0.0:
            print("delta == 0")
            temp_d = np.max(abs(new_col))
        print("tempd:", temp_d)
        r_deltas.append(temp_d)

        # print("outer", np.outer(new_col, new_col)/temp_d)
        # print("outer zonder", np.outer(new_col, new_col))
        # used_tubes.append((x_as, y_as))

        # --------- TUBES ---------
        print("takes:", y_as, ", ", x_as)
        tube_fiber = get_fiber(tensor, j=y_as, i=x_as)
        print("t:", tube_fiber)

        approx = np.zeros(len(tube_fiber))
        for i in range(rank):
            approx = np.add(approx, cols[i][y_as] * cols[i][x_as] * tubes[i] * (1.0 / c_deltas[i]) * (1.0 / r_deltas[i]))

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


def aca_matrix_x_vector(tensor, max_rank, start_tube=None, random_seed=None, to_cluster=False):
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

    # Generate samples for stopping criteria and better start.
    sample_indices, sample_values = generate_tube_samples(tensor)
    print(f"Sample indices: {sample_indices} \n with sample values {sample_values}")
    sample_size = len(sample_values)

    # If no start matrix is given, initialize a random one.
    if start_tube is None:
        max_sample = np.max(np.abs(sample_values))
        index_sample_max = np.where(np.abs(sample_values) == max_sample)[0][0]

        x_as = sample_indices[index_sample_max, 2]
        y_as = sample_indices[index_sample_max, 1]
        print(f"chosen tube: x={x_as}, y={y_as}")

    else:
        x_as = start_tube[0]
        y_as = start_tube[1]

    m_idx = [y_as, x_as]
    matrix_idx.append(m_idx)

    rank = 0
    # Select new skeletons until desired rank is reached.
    while rank < max_rank:

        print(f"--------------------- RANK {rank} ----------------------")
        # print(tensor[z_as])

        # --------- TUBES ---------
        y_as = m_idx[0]
        x_as = m_idx[1]
        tube_fiber = tensor[:, y_as, x_as]
        # print("t:", m_idx, "; ", tube_fiber)

        approx = np.zeros(len(tube_fiber))
        for i in range(rank):
            approx = np.add(approx, matrices[i][m_idx] * tubes[i] * (1.0 / t_deltas[i]))
        new_tube = np.subtract(tube_fiber, approx)

        print("t after:", new_tube)
        new_abs_tube = abs(new_tube)
        tube_without_previous = np.delete(new_abs_tube, tubes_idx, axis=0)
        max_val = np.max(tube_without_previous)
        z_as = np.where(new_abs_tube == max_val)[0][0]
        print(f"max val: {max_val} on Z pos: {z_as}")

        tubes.append(new_abs_tube)
        t_deltas.append(max_val)
        tubes_idx.append(z_as)

        # --------- MATRICES ---------
        matrix = tensor[z_as, :, :]
        print("matrix before", z_as, "; ", matrix)

        approx = np.zeros(matrix.shape)
        for i in range(rank):
            approx = np.add(approx, matrices[i] * tubes[i][z_as] * (1.0 / t_deltas[i]))
            # print("approx: \n", approx)
        new_matrix = np.subtract(matrix, approx)
        print("matrix after: \n", new_matrix)

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

        aca_norm = compare_aca_original(matrices, tubes, t_deltas, tensor)
        aca_matrix_norms.append(aca_norm)

        rank += 1

    if to_cluster:
        return matrices, tubes, m_deltas, t_deltas
    else:
        return aca_matrix_norms


def compare_aca_original(matrices, tubes, t_delta, original):
    t = np.zeros(original.shape)
    # Make whole tensor as sum of product of matrices and tube vectors
    # print("ts", tubes)
    for i in range(len(matrices)):
        matrix = matrices[i]
        # print("m", matrix)
        tube = np.divide(tubes[i], t_delta[i])
        t += np.einsum('i,jk->ijk', tube, matrix)
    print(t)
    # Fill diagonal of each frontal matrix in tensor with zeros
    for i in range(len(t)):
        np.fill_diagonal(t[i], 0)

    # Compare with original tensor
    difference = original-t
    # print("differ \n", difference)
    f_norm = np.linalg.norm(difference)
    original_norm = np.linalg.norm(original)
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
    reconstructed = np.abs(reconstruct_tensor(cp, symm=False))
    difference = np.subtract(original, reconstructed)
    # Calculate tensor norm (~Frobenius matrix norm)
    # f_norm2 = calc_norm(difference)
    f_norm = np.linalg.norm(difference)
    original_norm = np.linalg.norm(original)
    print("original norm = ", original_norm)
    print("fnorm =", f_norm)
    # original_norm2 = calc_norm(original)
    norm = f_norm/original_norm
    # norm2 = f_norm2/original_norm2
    return norm


def calc_norm(tensor):
    s = 0
    shape = tensor.shape
    for k in range(shape[0]):
        for j in range(shape[1]):
            for i in range(shape[2]):
                s += (tensor[k, j, i] * tensor[k, j, i])
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


def generate_samples(tensor):
    k, j, i = tensor.shape

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
