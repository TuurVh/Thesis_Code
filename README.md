# Thesis_Code
This repository holds all code used for my thesis about "Adaptieve tensor factorisaties om versneld tijdreeksen te clusteren". 
The implementation is a generalization of the Adaptive Cross Approximation (ACA) algorithm for matrices by M. Pede. This code van be found on his Github page [1]
and is described in the paper "Snel clusteren van tijdreeksen via lage-rang benaderingen" [2]. The data used throughout the thesis is from Decroos et al., [3].

### Add the dataset
To be able to construct the full tensor, the dataset from [3] needs to be represent in a folder called "data".
The following code will reconstruct the full tensor from this folder, and save it to "save_to".
```python
import numpy as np
import pandas as pd
import dtaidistance
import hdf_readtest
import h5py

path = "data/amie-kinect-data.hdf"
make_tensor(path, save_tensor="save_to)

```
### Use this tensor
To save time, we already created this tensor, which is stored as "full_tensor.npy" in the "tensors"-folder.
To create the low-rank approximation, the following code can be used:
```python
from ACA_T_matrix import aca_matrix_x_vector
from ACA_T_vectors import aca_tensor

path = "../tensors/full_tensor.npy"
big_t = np.load(path)

# Rank to choose
max_rank = 45

rows, columns, tubes, r_deltas, c_deltas = aca_tensor(big_t, max_rank, random_seed=None, to_cluster=True)
matrices, m_deltas, tubes = aca_matrix_x_vector(big_t, max_rank, start_matrix=None, random_seed=None, to_cluster=True)
```
The chosen rows, columns and tubes are returned for method 2 (vector method), and the matrices and tubes are returned for method 1 (matrix method).
The relative error can now be calculated with:
```python
# For matrix method:
rel_err = compare_aca_original(matrices, tubes, m_deltas, big_t)
# For vectors method:
matrices = preprocess_to_matrices(columns, rows, r_deltas)
rel_err = compare_aca_original(matrices, tubes, c_deltas, big_t)
```

## References
[1]: M. Pede. Fast-time-series-clustering, 2020. https://github.com/MathiasPede/Fast-Time-Series-Clustering.

[2]: M. Pede. Snel clusteren van tijdreeksen via lage-rang benaderingen. Master’s thesis, Faculteit Ingenieurswetenschappen, KU Leuven, Leuven, Belgium, 2020.

[3]: T. Decroos, K. Schutte, T. Beéck, B. Vanwanseele, and J. Davis. AMIE: Automatic Monitoring of Indoor Exercises: European Conference, ECML PKDD 2018, Dublin, Ireland, 
September 10-14, 2018, Proceedings, Part III, pages 424–439. 01 2019

## License
```
Copyright 2022-2023 KU Leuven

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
