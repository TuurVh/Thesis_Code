import random
import numpy as np
import tensorly as tl

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

    # Generate samples for better start
    sample_indices, sample_values = generate_tube_samples(tensor)
    # print(f"Sample indices: {sample_indices} \n with sample values {sample_values}")
    sample_size = len(sample_values)

    # If no start column is given, initialize a random one.
    if start_matrix is None:
        max_sample = np.max(np.abs(sample_values))
        index_sample_max = np.where(np.abs(sample_values) == max_sample)[0][0]

        z_as = sample_indices[index_sample_max, 0]

    else:
        z_as = start_matrix
    tubes_idx.append(z_as)

    restartable_samples = sample_values
    restartable_indices = sample_indices
    deleted_indices = np.array([], dtype=int)
    rank = 1
    # Select new skeletons until desired rank is reached.
    while rank < max_rank+1:
        # --------- MATRICES ---------
        # Calculate matrix from tensor
        matrix = tensor[z_as, :, :]

        # Use current approximation to find residual (new_matrix)
        approx = np.zeros(matrix.shape)
        for i in range(rank-1):
            approx = np.add(approx, matrices[i] * tubes[i][z_as] * (1.0 / m_deltas[i]))
        new_matrix = np.subtract(matrix, approx)

        new_abs_matrix = abs(new_matrix)
        # Setting previously used scores to zero to make sure they are not used twice
        for idx in matrix_idx:
            # symmetric matrix
            i, j = idx
            new_abs_matrix[i][j] = 0
            new_abs_matrix[j][i] = 0

        # Find max absolute value in the matrix, to use as delta
        m_idx = np.unravel_index(np.argmax(new_abs_matrix), new_matrix.shape)
        max_val = new_matrix[m_idx]

        # Store matrices, deltas and used index values
        matrices.append(new_matrix)
        m_deltas.append(max_val)
        matrix_idx.append(m_idx)

        y_as = m_idx[0]
        x_as = m_idx[1]

        # Delete the samples on the pivot row from the restartable samples
        in_chosen_tube = np.where(sample_indices[:, 0] == z_as)[0]
        in_chosen_matrix = np.where((sample_indices[:, 1] == y_as) & (sample_indices[:, 2] == x_as))[0]
        indices_in_samples = np.concatenate((in_chosen_tube, in_chosen_matrix))
        if deleted_indices.size == 0:
            deleted_indices = indices_in_samples
        else:
            deleted_indices = np.concatenate((deleted_indices, indices_in_samples), axis=0)
        restartable_samples = np.delete(sample_values, deleted_indices, axis=0)
        restartable_indices = np.delete(sample_indices, deleted_indices, axis=0)

        # --------- TUBES ---------
        # Calculate tube from tensor
        tube_fiber = tensor[:, y_as, x_as]

        # Use current approximation to find residual (new_tube)
        approx = np.zeros(len(tube_fiber))
        for i in range(rank-1):
            approx = np.add(approx, matrices[i][m_idx] * tubes[i] * (1.0 / m_deltas[i]))
        new_tube = np.subtract(tube_fiber, approx)

        # Remove previously chosen matrices from options for next matrix
        new_abs_tube = abs(new_tube)
        tube_without_previous = np.delete(new_abs_tube, tubes_idx, axis=0)
        # Find max value for next iteration
        max_val = np.max(tube_without_previous)
        z_as = np.where(new_abs_tube == max_val)[0][0]

        # Store tubes, deltas and used index values
        tubes.append(new_tube)
        t_deltas.append(max_val)
        tubes_idx.append(z_as)

        # Reevaluate samples
        # for s in range(sample_size):
        #     x_ = sample_indices[s, 1]
        #     y_ = sample_indices[s, 2]
        #     z_ = sample_indices[s, 0]
        #     temp_sample = sample_values[s]
        #     sample_values[s] = temp_sample - (matrices[rank-1][x_][y_] * tubes[rank-1][z_] * (1.0/m_deltas[rank-1]))

        # Find the maximum error on the samples
        if restartable_samples.size == 0:
            max_residu = 0
        else:
            max_residu = np.max(np.abs(restartable_samples))

        # Check whether the max of the row is smaller than the max residu from the samples, if so, switch
        if abs(max_val) < max_residu - 0.001:
            # Switch to the sample with the largest remaining value
            index_sample_max = np.where(np.abs(restartable_samples) == max_residu)[0][0]
            # If below is not commented, we update based on the max residu value (but makes it worse..)
            # z_as = restartable_indices[index_sample_max][0]

        # Reconstruction
        aca_norm = compare_aca_original(matrices, tubes, m_deltas, tensor)
        aca_matrix_norms.append(aca_norm)

        rank += 1

    if to_cluster:
        return matrices, m_deltas, tubes
    else:
        return aca_matrix_norms


def compare_aca_original(matrices, tubes, m_delta, original):
    t = np.zeros(original.shape)
    # Make whole tensor as sum of product of matrices and tube vectors
    for i in range(len(matrices)):
        matrix = np.divide(matrices[i], m_delta[i])
        tube = tubes[i]
        t += np.einsum('i,jk->ijk', tube, matrix)
    # Fill diagonal of each frontal matrix in tensor with zeros
    for i in range(len(t)):
        np.fill_diagonal(t[i], 0)

    # Compare with original tensor
    difference = original-t
    f_norm = np.linalg.norm(difference)
    original_norm = np.linalg.norm(original)
    norm = f_norm/original_norm
    return norm


def reconstruct_tensor(cp, zeros=True):
    reconstructed = tl.cp_tensor.cp_to_tensor(cp)

    # Make all elements on diagonal 0.
    if zeros:
        for i in range(len(reconstructed)):
            np.fill_diagonal(reconstructed[i], 0)

    # Round all elements in tensor for better overview
    # for i in range(len(reconstructed)):
    #     reconstructed[i] = np.matrix.round(reconstructed[i], decimals=2)
    return reconstructed


def compare_cp_with_full(cp, original):
    reconstructed = np.abs(reconstruct_tensor(cp, zeros=True))
    difference = np.subtract(original, reconstructed)

    # Calculate tensor norm (~Frobenius matrix norm)
    # f_norm2 = calc_norm(difference)
    f_norm = np.linalg.norm(difference)
    original_norm = np.linalg.norm(original)
    norm = f_norm/original_norm
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


def generate_matrix_samples(matrix):
    rs, cs = matrix.shape
    sample_idxs = np.zeros(shape=(rs, 2), dtype=int)
    sample_values = np.zeros(rs, dtype=float)

    for i in range(rs):
        x = i
        y = i
        while x == y:
            y = random.randint(0, cs-1)
        sample_idxs[i, 0] = x
        sample_idxs[i, 1] = y
        sample_values[i] = matrix[x][y]
    return sample_idxs, sample_values


def generate_tube_samples(tensor):
    zs, ys, xs = tensor.shape
    sample_indices = np.zeros(shape=(zs, 3), dtype=int)
    sample_values = np.zeros(zs, dtype=float)

    # Cycle over all horizontal slices, then take random (non-diagonal) element of that slice
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

