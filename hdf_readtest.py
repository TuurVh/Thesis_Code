import h5py
import pandas as pd
import numpy as np

def decode(array):
    return [x.decode() for x in array]

def partial_load(input_file, key, col):
    # Open filename in read mode
    handle = h5py.File(input_file, 'r')
    # Get columns and rows
    columns = decode(handle.get("{}/axis0".format(key))[:])
    rows = handle.get("{}/axis1".format(key))[:]

    # Find desired column and only read that column into a dataframe
    col_subset_idx = columns.index(col)
    matrix = handle.get("{}/block0_values".format(key))[:, col_subset_idx]
    df = pd.DataFrame(matrix, columns=[col], index=rows)

    return df


path = "data/amie-kinect-data.hdf"
subset = partial_load(path, 'skeleton_1', 'AnkleLeftX')
print(subset)
