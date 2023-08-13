import numpy as np
import pandas as pd
import dtaidistance
import openpyxl
from matplotlib import pyplot as plt
from scipy.stats import zscore
import time
import hdf_readtest
import h5py

np.set_printoptions(linewidth=np.inf)


def choose_skeletons_tensor(path, amount_ts, exercises=None, persons=None, save_tensor_to=None):
    overview = pd.read_hdf(path, key="overview")
    keys = overview["df_key"]

    handle = h5py.File(path, 'r')
    column_names = hdf_readtest.decode(handle.get("{}/axis0".format(keys[0]))[:])
    columns = column_names[0:amount_ts]
    # columns = ["AnkleLeftX", "AnkleLeftY", "AnkleLeftZ", "AnkleRightX", "AnkleRightY", "AnkleRightZ", "HipLeftX",
    #            "HipLeftY", "HipLeftZ", "HipRightX", "HipRightY", "HipRightZ",
    #            "ShoulderLeftX", "ShoulderLeftY", "ShoulderLeftZ",
    #            "ShoulderRightX", "ShoulderRightY", "ShoulderRightZ"
    #            ]

    skeletons = []
    if exercises is not None:
        for idx, elem in overview["exercise"].items():
            if elem in exercises:
                skeleton = keys[idx]
                if persons is None:
                    skeletons.append(skeleton)
                else:
                    person = overview["person"][idx]
                    if person in persons:
                        skeletons.append(skeleton)
    if persons is not None:
        for idx, elem in overview["person"].items():
            if elem in persons:
                skeleton = keys[idx]
                skeletons.append(skeleton)

    tensor = []
    # Filter out only the first correct execution:
    # vids = [skeletons[i] for i in range(len(skeletons)) if (i+1) % 6 == 0]
    vids = skeletons

    for c in columns:
        matrix = np.zeros((len(vids), len(vids)))
        for i in range(len(vids)):
            print("c.i=", c, ".", i)
            # Find the persons in the DB and take out the needed time series
            person1 = vids[i]
            p_df = hdf_readtest.partial_load(path, key=person1, col=c)
            ts_p1 = np.concatenate(p_df.to_numpy())
            # ts_p1 = zscore(ts_p1)
            for j in range(i+1, len(vids)):
                person2 = vids[j]
                p_df = hdf_readtest.partial_load(path, key=person2, col=c)
                ts_p2 = np.concatenate(p_df.to_numpy())
                # ts_p2 = zscore(ts_p2)

                # Calculate the DTW-distance between the two TS and store in distance matrix
                temp = dtaidistance.dtw.distance(ts_p1, ts_p2, use_c=True)
                matrix[i, j] = temp
                matrix[j, i] = temp

        tensor.append(matrix)
    tensor = np.asarray(tensor)
    if save_tensor_to is not None:
        np.save(save_tensor_to, tensor)
    return tensor


def make_short_tensor(path, amount_columns, vids, save_tensor=False):
    overview = pd.read_hdf(path, key="overview")
    persons = overview["df_key"]

    handle = h5py.File(path, 'r')
    column_names = hdf_readtest.decode(handle.get("{}/axis0".format(persons[0]))[:])
    columns = column_names[0:amount_columns]

    tensor = []

    for c in columns:
        matrix = np.zeros((vids, vids))
        for i in range(vids):
            print("c.i=", c, ".", i)
            for j in range(i+1, vids):
                # Find the persons in the DB and take out the needed time series
                person1 = persons[i]
                p_df = hdf_readtest.partial_load(path, key=person1, col=c)
                ts_p1 = np.concatenate(p_df.to_numpy())
                ts_p1 = zscore(ts_p1)

                person2 = persons[j]
                p_df = hdf_readtest.partial_load(path, key=person2, col=c)
                ts_p2 = np.concatenate(p_df.to_numpy())
                ts_p2 = zscore(ts_p2)

                # Calculate the DTW-distance between the two TS and store in distance matrix
                temp = dtaidistance.dtw.distance(ts_p1, ts_p2, use_c=True)
                matrix[i, j] = temp
                matrix[j, i] = temp

        tensor.append(matrix)
    tensor = np.asarray(tensor)
    if save_tensor:
        np.save("tensor", tensor)
    np.set_printoptions(linewidth=np.inf)
    return tensor


def z_normalize_time_series(time_series):
    mean = np.mean(time_series)
    std_dev = np.std(time_series)
    z_normalized_series = (time_series - mean) / std_dev
    return z_normalized_series


def make_tensor(path, save_tensor=None):
    overview = pd.read_hdf(path, key="overview")
    persons = overview["df_key"]
    length = len(persons)

    handle = h5py.File(path, 'r')
    columns = hdf_readtest.decode(handle.get("{}/axis0".format(persons[0]))[:])
    tensor = []

    for c in columns:
        matrix = np.zeros((length, length))
        for i in range(length):
            print("c.i=", c, ".", i, "van de", length)
            # Find the persons in the DB and take out the needed time series
            person1 = persons[i]
            p_df = hdf_readtest.partial_load(path, key=person1, col=c)
            ts_p1 = np.concatenate(p_df.to_numpy())
            for j in range(i+1, length):
                person2 = persons[j]
                p_df = hdf_readtest.partial_load(path, key=person2, col=c)
                ts_p2 = np.concatenate(p_df.to_numpy())
                # Calculate the DTW-distance between the two TS and store in distance matrix
                temp = dtaidistance.dtw.distance(ts_p1, ts_p2, use_c=True)
                matrix[i, j] = temp
                matrix[j, i] = temp
        tensor.append(matrix)
    tensor = np.asarray(tensor)
    if save_tensor is not None:
        np.save(save_tensor, tensor)
    return tensor


def save_overview(path):
    overview = pd.read_hdf(path, key="overview")
    df = pd.DataFrame(overview)
    df.to_csv("overview.csv")

path = "data/amie-kinect-data.hdf"

# start_time = time.time()
# T = make_tensor(path, save_tensor="tensors/notfull_tensor")
# end_time = time.time()
# duration = end_time - start_time
# print("duration of one execution = ", duration // 60, "minutes and ", duration % 60, 'seconds.')

# T = make_short_tensor(path, amount_columns=9, vids=40, save_tensor=True)
# for a in range(50, 76, 10):
#     save = "tensors/person2-3-5_all_ex_" + str(a) + "ts"
T = choose_skeletons_tensor(path, amount_ts=75, persons=["person2", "person3"], save_tensor_to="tensors/person2-3-ALL-normalized")
# T = choose_skeletons_tensor(path, amount_ts=75, exercises=["squat"], save_tensor_to="tensors/all_p_squat")
print("shape: ", T.shape)
print("tensor: ", T)
