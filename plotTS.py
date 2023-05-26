import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import hdf_readtest
from matplotlib.pyplot import savefig


def plot_ts(path):
    overview = pd.read_hdf(path, key="overview")
    print(overview)
    p_df = hdf_readtest.partial_load(path, key="skeleton_3", col="HeadY")
    print(p_df)
    plt.rcParams['figure.figsize'] = [10, 4]
    # Deletes the upper and right edges (for poster)
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    plt.plot(p_df, lw=3)
    plt.savefig('demo.png', format='png', transparent=True)
    plt.show()


path = "data/amie-kinect-data.hdf"

plot_ts(path)
