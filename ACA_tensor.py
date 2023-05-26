import random
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
import plotting
import clustering
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering


np.set_printoptions(linewidth=np.inf)


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
        print("x_used", x_used)
        print("y_used", y_used)
        print("z_used", z_used)

        print(f"--------------------- RANK {rank} ----------------------")
        # print(tensor[z_as])
        #  --------- TUBES ---------
        tube_fiber = tensor[:, y_as, x_as]
        print("t:", tube_fiber)

        approx = np.zeros(len(tube_fiber))
        for i in range(rank):
            approx = np.add(approx, matrices[i][m_idx] * tubes[i] * (1.0 / m_deltas[i]))
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

        k_cols = np.vstack(k_cols)
        temp = [np.divide(k_row, rd) for k_row, rd in zip(k_rows, r_ds)]
        k_rows = (np.vstack(temp))

        matrix = np.dot(k_cols.T, k_rows)
        matrices.append(matrix)
        print("matrix", matrix.shape)
        new_abs_matrix = abs(matrix)
        for idx in matrix_idx:
            # symmetric matrix
            i, j = idx
            new_abs_matrix[i][j] = 0
            new_abs_matrix[j][i] = 0

        m_idx = np.unravel_index(np.argmax(new_abs_matrix), matrix.shape)
        max_val = matrix[m_idx]
        print(f"max val: {max_val} on matrix pos: {m_idx}")
        x_used.append(m_idx[1])
        y_used.append(m_idx[0])
        matrix_idx.append(m_idx)
        m_deltas.append(max_val)

        aca_norm = compare_aca_original(matrices, tubes, t_deltas, tensor)
        aca_k_hat_norms.append(aca_norm)

        cols.append(new_col)
        c_deltas.append(max_val_y)
        rows.append(new_row)
        r_deltas.append(max_val_x)
        rank += 1
    return aca_k_hat_norms

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
        print("x_used", x_used)
        print("y_used", y_used)
        print("z_used", z_used)

        print(f"--------------------- RANK {rank} ----------------------")
        # print(tensor[z_as])
        #  --------- TUBES ---------
        tube_fiber = tensor[:, y_as, x_as]
        print("t:", tube_fiber)

        approx = np.zeros(len(tube_fiber))
        for i in range(rank):
            approx = np.add(approx, matrices[i][m_idx] * tubes[i] * (1.0 / m_deltas[i]))
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
        new_abs_matrix = abs(m_tot)
        for idx in matrix_idx:
            # symmetric matrix
            i, j = idx
            new_abs_matrix[i][j] = 0
            new_abs_matrix[j][i] = 0

        m_idx = np.unravel_index(np.argmax(new_abs_matrix), m_tot.shape)
        max_val = m_tot[m_idx]
        print(f"max val: {max_val} on matrix pos: {m_idx}")
        x_used.append(m_idx[1])
        y_used.append(m_idx[0])
        matrix_idx.append(m_idx)
        m_deltas.append(max_val)

        aca_norm = compare_aca_original(matrices, tubes, t_deltas, tensor)
        aca_k_hat_norms.append(aca_norm)

        cols.append(new_col)
        c_deltas.append(max_val_y)
        rows.append(new_row)
        r_deltas.append(max_val_x)
        rank += 1
    return aca_k_hat_norms



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
# optie voor random seed en begin kolom
# MET CHECK VOOR DUBBELE FIBERS
def aca_tensor(tensor, max_rank, start_col=None, random_seed=None):
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

    x_used.append(x_as)
    z_used.append(z_as)

    rank = 0
    # Select new skeletons until desired rank is reached.
    while rank < max_rank:

        print(f"--------------------- RANK {rank} ----------------------")
        # print(tensor[z_as])

        # --------- COLUMNS ---------
        col_fiber = tensor[z_as, :, x_as]
        print("col before", col_fiber)

        approx = np.zeros(len(col_fiber))
        for i in range(rank):
            approx = np.add(approx, cols[i] * rows[i][x_as] * tubes[i][z_as] * (1.0 / r_deltas[i]) * (1.0 / t_deltas[i]))
        new_col = np.subtract(col_fiber, approx)
        print("col after", new_col)

        new_abs_col = abs(new_col)
        previous = [i for i, item in enumerate(z_used[0:len(z_used)-1]) if item == z_as]
        # print("previous y", y_used)
        to_delete = [y_used[p] for p in previous]
        # print("y to delete", to_delete)
        col_without_previous = np.delete(new_abs_col, to_delete, axis=0)
        max_val = np.max(col_without_previous)

        y_as = np.where(abs(new_col) == max_val)[0][0]
        print(f"max val: {new_col[y_as]} on Y pos: {y_as}")
        cols.append(new_col)
        c_deltas.append(new_col[y_as])
        y_used.append(y_as)

        # --------- ROWS ---------
        row_fiber = tensor[z_as, y_as, :]
        print("r before", row_fiber)

        approx = np.zeros(len(row_fiber))
        for i in range(rank):
            approx = np.add(approx, cols[i][y_as] * rows[i] * tubes[i][z_as] * (1.0 / c_deltas[i]) * (1.0 / t_deltas[i]))
        new_row = np.subtract(row_fiber, approx)
        print("r after", new_row)

        new_abs_row = abs(new_row)
        previous = [i for i, item in enumerate(y_used[0:len(y_used)-1]) if item == y_as]
        # print("prevous x", x_used)
        to_delete = [x_used[p+1] for p in previous]
        # print("x to delete", to_delete)
        row_without_previous = np.delete(new_abs_row, to_delete, axis=0)
        max_val = np.max(row_without_previous)

        x_as = np.where(abs(new_row) == max_val)[0][0]
        print(f"max val: {new_row[x_as]} on X pos: {x_as}")
        rows.append(new_row)
        r_deltas.append(new_row[x_as])
        x_used.append(x_as)

        # --------- TUBES ---------
        tube_fiber = tensor[:, y_as, x_as]
        print("t:", tube_fiber)

        approx = np.zeros(len(tube_fiber))
        for i in range(rank):
            approx = np.add(approx, cols[i][y_as] * rows[i][x_as] * tubes[i] * (1.0 / c_deltas[i]) * (1.0 / r_deltas[i]))
        new_tube = np.subtract(tube_fiber, approx)

        print("t after:", new_tube)

        new_abs_tube = abs(new_tube)
        previous = [i for i, item in enumerate(x_used[0:len(x_used)-1]) if item == x_as]
        # print("previous z", z_used)
        to_delete = [z_used[p] for p in previous]
        # print("z to delete", to_delete)
        tube_without_previous = np.delete(new_abs_tube, to_delete, axis=0)
        max_val = np.max(tube_without_previous)

        z_as = np.where(new_abs_tube == max_val)[0][0]
        print(f"max val: {new_tube[z_as]} on Z pos: {z_as}")

        tubes.append(new_tube)
        t_deltas.append(new_tube[z_as])
        z_used.append(z_as)

        rank += 1
    return cols, rows, tubes, c_deltas, r_deltas, t_deltas


def compare_aca_original(matrices, tubes, tube_delta, original):
    t = np.zeros(original.shape)
    # Make whole tensor as sum of product of matrices and tube vectors
    for i in range(len(matrices)):
        matrix = matrices[i]
        tube = np.divide(tubes[i], tube_delta[i])
        t += np.einsum('i,jk->ijk', tube, matrix)

    # Fill diagonal of each frontal matrix in tensor with zeros
    for i in range(len(t)):
        np.fill_diagonal(t[i], 0)
    # print("einsum \n", t)

    # Compare with original tensor
    difference = original-t
    norm = np.linalg.norm(difference)

    return norm


def compare_aca_k_original(cols, rows, tubes, row_delta, tube_delta, original):
    t = np.zeros(original.shape)
    # Make whole tensor as sum of skeletons
    for i in range(len(cols)):
        col = cols[i]
        row = np.divide(rows[i], row_delta[i])
        # print(row)
        # col = np.divide(cols[i], col_delta[i])
        tube = np.divide(tubes[i], tube_delta[i])
        matrix = np.dot(col.T, row)
        t += np.einsum('i,jk->ijk', tube, matrix)

    # Fill diagonal of each frontal matrix in tensor with zeros
    for i in range(len(t)):
        np.fill_diagonal(t[i], 0)
    # print("einsum \n", t)

    # Compare with original tensor
    difference = original-t
    norm = np.linalg.norm(difference)

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
    combined_fs = [f_ts, f_rs, f_cs]
    weights = np.ones(len(row_delta))
    for w in range(len(weights)):
        weights[w] = (1/row_delta[w]) * (1/tube_delta[w])
    return weights, combined_fs


def compare_cp_with_full(cp, original):
    reconstructed = tl.cp_tensor.cp_to_tensor(cp)

    # Make all elements on diagonal 0.
    for i in range(len(reconstructed)):
        np.fill_diagonal(reconstructed[i], 0)

    difference = original-reconstructed
    # Calculate tensor norm (~Frobenius matrix norm)
    norm = np.linalg.norm(difference)
    return norm


def one_rank(path, rank):
    big_t = np.load(path)
    # ms, ts, md, td = aca_matrix_x_vector(big_t, max_rank=rank, start_matrix=None, random_seed=2)
    # aca_norm = compare_aca_original(ms, ts, td, big_t)

    cs, rs, ts, cd, rd, td = aca_k_vectors(big_t, max_rank=rank, k_hat=3, random_seed=2)
    aca_norm = compare_aca_k_original(cs, rs, ts, rd, td, big_t)
    # factors_aca = aca_as_cp(cs, rs, ts, rd, td)
    # aca_norm = compare_cp_with_full(cp=factors_aca, original=big_t)

    factors_cp = parafac(big_t, rank=rank, normalize_factors=False)
    cp_norm = compare_cp_with_full(cp=factors_cp, original=big_t)

    print("diff A & ACA", aca_norm)
    # print("T", big_t)
    print("diff A & CP", cp_norm)
    # print("rec", reconstructed_tensor)


def more_ranks(path, ranks, plot=True):
    big_t = np.load(path)
    seed = 1
    aca_vects_norms = []
    aca_k_hat_norms = []
    aca_matrix_norms = []
    cp_norms = []

    # for loop is sub-optimaal voor dit! altijd zelfde herberekenen => beter = telkens 1 verdere stap i/d while loop
    # van de main aca loop!!
    for rank in ranks:

        cs, rs, ts, cd, rd, td = aca_tensor(big_t, max_rank=rank, start_col=None, random_seed=seed)
        factors_aca = aca_as_cp(cs, rs, ts, rd, td)
        aca_norm = compare_cp_with_full(cp=factors_aca, original=big_t)
        aca_vects_norms.append(aca_norm)

        # cs, rs, ts, cd, rd, td = aca_k_vectors(big_t, max_rank=rank, k_hat=3, start_col=None, random_seed=seed)
        # aca_norm = compare_aca_k_original(cs, rs, ts, rd, td, big_t)
        # aca_k_hat_norms.append(aca_norm)
        aca_k_hat_norms.append(0)

        ms, ts, md, td = aca_matrix_x_vector(big_t, max_rank=rank, start_matrix=None, random_seed=seed)
        aca_norm = compare_aca_original(ms, ts, td, big_t)
        aca_matrix_norms.append(aca_norm)

        factors_cp = parafac(big_t, rank=rank, normalize_factors=False)
        cp_norm = compare_cp_with_full(cp=factors_cp, original=big_t)
        cp_norms.append(cp_norm)

        print("diff A & ACA", aca_norm)
        # print("T", big_t)
        print("diff A & CP", cp_norm)
        # print("rec", reconstructed_tensor)

    # PLOTTING
    if plot:
        print("ACA Matrix:", aca_matrix_norms)
        print("ACA k_hat:", aca_k_hat_norms)
        print("ACA vectors: ", aca_vects_norms)
        print("CP:", cp_norms)
        plotting.plot_norms_aca_cp(aca_matrix_norms, aca_k_hat_norms, aca_vects_norms, cp_norms, ranks)
        k_hat = len(cs[0])
        plotting.plot_amount_calcs(big_t, k_hat, ranks)

def main():
    path = "tensors/person2-3-5_all_ex_50ts.npy"

    # To calculate the rank of one tensor, given the path to the tensor.
    # one_rank(path, 3)

    # ranks given as a range, will do ACA and CP for all the elements in range.
    ranks = range(1, 50)
    more_ranks(path, ranks, plot=True)


if __name__ == "__main__":
    main()


# STARTED W/ CLUSTERING
# labels = clustering.cluster_three_dims(rs, cs, ts)
# print(labels)

# Visualizing the clustering
# plt.scatter(X_principal['P1'], X_principal['P2'],
#            c = SpectralClustering(n_clusters = 2, affinity ='rbf') .fit_predict(X_principal), cmap=plt.cm.winter)
# plt.show()
