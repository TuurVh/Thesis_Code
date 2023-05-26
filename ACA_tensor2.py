import random
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

np.set_printoptions(linewidth=np.inf)

def aca_matrix_x_vector(tensor, max_rank, start_matrix=None, random_seed=None):
    # The already chosen matrices and tubes are stored.
    matrices = []
    tubes = []
    # The found deltas are stored.
    m_deltas = []
    t_deltas = []
    # The index of matrices and tubes are stored to make sure they are not used twice.
    matrix_idx = []
    tubes_idx = []

    # Def shape of tensor
    shape = tensor.shape  # shape = (z, y, x)

    # If a random seed is given, use this.
    if random_seed is not None:
        random.seed(random_seed)

    # If no start column is given, initialize a random one.
    if start_matrix is None:
        z_as = random.randint(0, shape[0]-1)
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
        # print("matrix before", matrix)

        approx = np.zeros(matrix.shape)
        for i in range(rank):
            approx = np.add(approx, matrices[i] * tubes[i][z_as] * (1.0 / t_deltas[i]))
        new_matrix = np.subtract(matrix, approx)
        # print("col after", new_matrix)

        new_abs_matrix = abs(new_matrix)
        # Setting previously used scores to zero to make sure they are not used twice
        for idx in matrix_idx:
            # symmetric matrix
            i, j = idx
            new_abs_matrix[i][j] = 0
            new_abs_matrix[j][i] = 0

        m_idx = np.unravel_index(np.argmax(new_abs_matrix), new_matrix.shape)
        max_val = matrix[m_idx]
        print(f"max val: {max_val} on Y pos: {m_idx}")

        matrices.append(new_matrix)
        m_deltas.append(max_val)
        matrix_idx.append(m_idx)

        # --------- TUBES ---------
        y_as = m_idx[0]
        x_as = m_idx[1]
        tube_fiber = tensor[:, y_as, x_as]
        print("t:", tube_fiber)

        approx = np.zeros(len(tube_fiber))
        for i in range(rank):
            approx = np.add(approx, matrices[i][m_idx] * tubes[i] * (1.0 / m_deltas[i]))
        new_tube = np.subtract(tube_fiber, approx)

        print("t after:", new_tube)
        new_abs_tube = abs(new_tube)
        tube_without_previous = np.delete(new_abs_tube, tubes_idx, axis=0)
        max_val = np.max(tube_without_previous)
        z_as = np.where(new_abs_tube == max_val)[0][0]
        print(f"max val: {(new_tube[z_as])} on Z pos: {z_as}")

        tubes.append(new_tube)
        t_deltas.append((new_tube[z_as]))
        tubes_idx.append(z_as)

        rank += 1
    return matrices, tubes, m_deltas, t_deltas


# Berekent ACA decompositie van een tensor, zonder de volledige afstandstensor te moeten opstellen
# Tensor zit in een bepaalde file, functie "getDTW..." om de dtw van bepaalde row/col te berekenen
# Eigenlijk onzin want bij gebruik kleinere tensor gebruik van bepaalde skeletten/time series
# ==> Eerste implementatie met gebruik tensor.
# optie voor random seed en begin rij / kolom
def aca_tensor(tensor, max_rank, start_col=None, random_seed=None):
    # test for tensor multiplication
    # mat2 = np.array([[1], [2], [3]])
    # mat1 = [1, 3, 1]
    # mat = mat1 * mat2
    # print(f"mat: {mat}")
    # test = mat * [[[1]], [[2]]]
    # print(f"test: {test}")

    rows = []
    cols = []
    tubes = []
    r_deltas = []
    c_deltas = []
    t_deltas = []
    r_indices = []
    c_indices = []
    t_indices = []

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

    rank = 0
    # Select new skeletons until desired rank is reached.
    while rank < max_rank:

        print(f"--------------------- RANK {rank} ----------------------")
        print(tensor[z_as])

        # --------- COLUMNS ---------
        col_fiber = tensor[z_as, :, x_as]
        t_indices.append(z_as)
        print("col before", col_fiber)

        approx = np.zeros(len(col_fiber))
        # for i in range(rank):
        #     approx = np.add(approx, cols[i] * rows[i][x_as] * tubes[i][z_as] * (1.0 / r_deltas[i]) * (1.0 / t_deltas[i]))
        new_col = np.subtract(col_fiber, approx)

        print("col after", new_col)

        max_val = np.max(abs(new_col))
        y_as = np.where(abs(new_col) == max_val)[0][0]
        print(f"max val: {max_val} on Y pos: {y_as}")
        cols.append(new_col)
        c_deltas.append(max_val)

        # --------- ROWS ---------
        row_fiber = tensor[z_as, y_as, :]
        c_indices.append(y_as)
        print("r before", row_fiber)

        approx = np.zeros(len(row_fiber))
        # for i in range(rank):
        #     approx = np.add(approx, cols[i][y_as] * rows[i] * tubes[i][z_as] * (1.0 / c_deltas[i]) * (1.0 / t_deltas[i]))
        new_row = np.subtract(row_fiber, approx)
        print("r after", new_row)

        max_val = np.max(abs(new_row))
        x_as = np.where(abs(new_row) == max_val)[0][0]
        print(f"max val: {max_val} on X pos: {x_as}")
        rows.append(new_row)
        r_deltas.append(max_val)

        # --------- TUBES ---------
        tube_fiber = tensor[:, y_as, x_as]
        r_indices.append(x_as)
        print("t:", tube_fiber)

        approx = np.zeros(len(tube_fiber))
        # for i in range(rank):
        #     approx = np.add(approx, cols[i][y_as] * rows[i][x_as] * tubes[i] * (1.0 / c_deltas[i]) * (1.0 / r_deltas[i]))
        new_tube = np.subtract(tube_fiber, approx)

        print("t after:", new_tube)
        new_abs_tube = abs(new_tube)
        tube_without_previous = np.delete(new_abs_tube, t_indices, axis=0)
        max_val = np.max(tube_without_previous)
        z_as = np.where(new_abs_tube == max_val)[0][0]
        print(f"max val: {max_val} on Z pos: {z_as}")
        tubes.append(new_tube)
        t_deltas.append(max_val)

        rank += 1
    return cols, rows, tubes, c_deltas, r_deltas, t_deltas


def compare_aca_original(cols, rows, tubes, col_delta, row_delta, tube_delta, original):
    t = np.zeros(original.shape)
    # Make whole tensor as sum of skeletons
    for i in range(len(cols)):
        row = rows[i]
        col = cols[i]
        tube = tubes[i]
        # print(rows[i])
        # print(row_delta)
        row = np.divide(rows[i], row_delta[i])
        # print(row)
        # col = np.divide(cols[i], col_delta[i])
        tube = np.divide(tubes[i], tube_delta[i])
        t += np.einsum('i,j,k->ijk', tube, col, row)

    # Fill diagonal of each frontal matrix in tensor with zeros
    for i in range(len(t)):
        np.fill_diagonal(t[i], 0)
    print("einsum \n", t)

    # Compare with original tensor
    diff = original-t
    return diff


# big_t = np.load("tensor3p3times1ex.npy")

# big_t = np.array([[[0, 1],
#                    [3, 0]],
#                   [[0, 3],
#                    [2, 0]]])

big_t = np.array([[[0, 1, 2],
                   [3, 0, 4],
                   [2, 1, 0]],
                  [[0, 3, 2],
                   [2, 0, 1],
                   [1, 1, 0]],
                  [[0, 2, 2],
                   [1, 0, 3],
                   [1, 3, 0]]])


cs, rs, ts, cd, rd, td = aca_tensor(big_t, max_rank=1, start_col=None, random_seed=0)
print("cs", cs)
print("rs", rs)
print("ts", ts)

factors = parafac(big_t, rank=1, normalize_factors=False)
reconstructed_tensor = tl.cp_tensor.cp_to_tensor(factors)
print("factors", factors[1])

# Make all elements on diagonal 0.
for i in range(len(reconstructed_tensor)):
    np.fill_diagonal(reconstructed_tensor[i], 0)

print("diff", np.linalg.norm(compare_aca_original(cs, rs, ts, cd, rd, td, big_t)))
# print("result:", compare_aca_original(cs, rs, ts, cd, rd, td, big_t))
print("T", big_t)
print("diff 2", np.linalg.norm(big_t - reconstructed_tensor))
print("rec", reconstructed_tensor)