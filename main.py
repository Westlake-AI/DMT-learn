import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dmt import DMT_DV
from scipy.io import mmread


def read_mtx(filename, dtype='int32'):
    x = mmread(filename).astype(dtype)
    return x

# # Load single-batch dataset
# mtx = './dmt/data/cd14_monocyte_erythrocyte.mtx'
# data = read_mtx(mtx)
# data = np.asarray(data.transpose().todense())
# label = pd.read_csv('dmt/data/cd14_monocyte_erythrocyte_celltype.tsv', sep='\t', header=None).values
# label_id, label_id_set = pd.factorize(label.reshape(-1))

# # Perform DMT_DV
# dmt_dv = DMT_DV(manifold="Euclidean", nu=0.005, augNearRate=10, batchRate=0) # "Euclidean", "PoincareBall", "Hyperboloid"
# X_dmt_dv = dmt_dv.fit_transform(data)

# # Load multiple-batches dataset
mtx = './dmt/data/uc_stromal.mtx'
data = read_mtx(mtx)
data = np.asarray(data.transpose().todense())
label = pd.read_csv('dmt/data/uc_stromal_celltype.tsv', sep='\t', header=None).values
label_id, label_id_set = pd.factorize(label.reshape(-1))
label_batch_p = pd.read_csv('dmt/data/uc_stromal_batch_patient.tsv', header=None).values
label_batch_h = pd.read_csv('dmt/data/uc_stromal_batch_health.tsv', header=None).values
label_batch = np.asarray(np.concat([label_batch_p, label_batch_h], axis=1))

# Perform DMT
dmt_dv = DMT_DV(manifold="Euclidean", nu=0.01, augNearRate=100000, batchRate=1, lr=0.005, batch_size=2000, epochs=300)
# X_dmt_dv = dmt_dv.fit_transform(data, label_batch)

# # Plot the result
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_dmt_dv[:, 0], X_dmt_dv[:, 1], c=label_id, cmap='viridis', s=10)

# # Create legend
# legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
# plt.gca().add_artist(legend1)  # Add the legend to the current axes

# plt.title('DMT-DV visualization of Single Cell Dataset')
# plt.xlabel('DMT-DV Component 1')
# plt.ylabel('DMT-DV Component 2')
# plt.savefig('dmt_dv.png')

dmt_dv.compare(data, label_batch, "comparison.png")
