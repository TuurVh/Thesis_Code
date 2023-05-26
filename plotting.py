import matplotlib.pyplot as plt

def plot_norms_aca_cp(amount_vects, aca_m=None, aca_k=None, aca_v=None, cp_norms=None):
    amount_vects = range(1, amount_vects+1)
    num_cols = 0
    plt.yscale('log')
    if aca_m is not None:
        plt.plot(amount_vects, aca_m, label='A - ACA_matrix')
        num_cols += 1
    if aca_k is not None:
        plt.plot(amount_vects, aca_k, label='A - ACA_k_hat')
        num_cols += 1
    if aca_v is not None:
        plt.plot(amount_vects, aca_v, label='A - ACA_vectors')
        num_cols += 1
    if cp_norms is not None:
        plt.plot(amount_vects, cp_norms, label='A - CP')
        num_cols += 1

    plt.xlabel("# vectoren")
    plt.ylabel("|| Tensor - Benadering ||")

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=num_cols)
    plt.show()


def plot_amount_calcs(tensor, k_hat, rank):
    print("k_hat", k_hat)
    # tensor shape = i x i x j
    shape = tensor.shape
    print("shape", shape)
    amount_vectors = rank
    ranks = range(0, rank)

    # aca met vectoren: per iteratie i + i + j
    aca_v_calculations = shape[0] + shape[1] + shape[2]
    aca_v = [aca_v_calculations]*amount_vectors
    aca_v_tot = [(i+1)*aca_v[i] for i in range(len(aca_v))]

    # aca met k_hat vectoren: per iteratie (k_hat * i) + (k_hat * i) + j
    aca_k_calculations = shape[0] + k_hat*shape[1] + k_hat*shape[2]
    aca_k = [aca_k_calculations]*amount_vectors
    aca_k_tot = [(i+1)*aca_k[i] for i in range(len(aca_k))]

    # aca met matrix * vector: per iteratie (i * i) + j
    aca_m_calculations = shape[0] + (shape[1] * shape[2])
    aca_m = [aca_m_calculations]*amount_vectors
    aca_m_tot = [(i+1)*aca_m[i] for i in range(len(aca_m))]

    # CP: heeft volledige tensor nodig: i * i * j
    cp_calculations = shape[0] * shape[1] * shape[2]
    cp = [cp_calculations]*amount_vectors

    print("vs:", aca_v_tot)
    print("ks:", aca_k_tot)
    print("ms:", aca_m_tot)
    print("cp:", cp)

    plt.plot(ranks, aca_m_tot, label='ACA_matrix')
    plt.plot(ranks, aca_k_tot, label='ACA_k_hat')
    plt.plot(ranks, aca_v_tot, label='ACA_vectors')
    # plt.plot(ranks, cp, label='CP')

    plt.xlabel("# vectoren")
    plt.ylabel("# DTW berekeningen")
    # y-as exponentieel
    # plt.yscale('symlog')

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
    plt.show()



