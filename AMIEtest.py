import pandas as pd
from pandas import HDFStore

# Find the lengths of all the TS in AMIE dataset
def ts_lengths(path):
    overview = pd.read_hdf(path, key="overview")
    print(overview)
    persons = overview["df_key"]
    for p in persons:
        print(p)
        ts = pd.read_hdf(path, key=p)
        length = len(ts)
        if length <= 1100:
            print(" ----------- Short -------- ", length)
        else:
            print(length)

# Print all elements in the overview of AMIE set
def AMIE_overview(path):
    overview = pd.read_hdf(path, key="overview")
    amount = len(overview["df_key"])
    for o in range(amount):
        print(overview["df_key"][o], overview["execution_type"][o], overview["exercise"][o], overview["person"][o])


def skeleton_to_csv(path, skeleton):
    skel = pd.read_hdf(path, key=skeleton)
    file_out = skeleton + ".csv"
    skel.to_csv(file_out)


path = "data/amie-kinect-data.hdf"
print(pd.read_hdf(path, key="skeleton_1"))
ts_lengths(path)
AMIE_overview(path)
skeleton_to_csv(path, "skeleton_1")
