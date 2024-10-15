
# EVNet: An Explainable Deep Network for Dimension Reduction (EVNet)

The code includes the following modules:
* Training
* Inference
* Comparison with t-SNE and UMAP


## Requirements

* torch==2.3.1
* torchaudio>=2.3.1
* torchvision>=0.18.1
* pytorch-lightning==2.4.0

## Installation
Install from source:
```bash
pip install -e .
```
Install from PyPI:
```bash
pip install dmtev-learn
```

## Running the code
Use the following code to fit the model to the dataset and visualize the results.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from dmtev import DMTEV

# Load sample dataset
iris = load_iris()
X = iris.data
y = iris.target

# Perform t-SNE
dmt = DMTEV(num_fea_aim=100)
X_dmt = dmt.fit_transform(X)

# Plot the result
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_dmt[:, 0], X_dmt[:, 1], c=y, cmap='viridis')

# Create legend
legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
plt.gca().add_artist(legend1)  # Add the legend to the current axes

plt.title('DMT-EV visualization of Iris dataset')
plt.xlabel('DMT-EV Component 1')
plt.ylabel('DMT-EV Component 2')
plt.savefig('dmt.png')
```
If you want to compare the results with other dimension reduction methods(t-SNE, UMAP), you can use the following code:
```python
dmt.compare(X, "comparison.png")
```

## Cite the paper

```
@article{zang2023evnet,
  title={Evnet: An explainable deep network for dimension reduction},
  author={Zang, Zelin and Cheng, Shenghui and Lu, Linyan and Xia, Hanchen and Li, Liangyu and Sun, Yaoting and Xu, Yongjie and Shang, Lei and Sun, Baigui and Li, Stan Z},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2023},
  publisher={IEEE}
}
```



## License

EVNet is released under the MIT license.
