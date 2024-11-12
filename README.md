# DMTPRL: Deep Manifold Transformation for Protein Representation Learning

The code includes the following modules:
* Training
* Inference

## Environments for Downstream Tasks
**Main dependencies**
- python>=3.7
- pytorch>=1.9
- transformer>=4.5.1+
- lmdb
- tape_proteins
- scikit-multilearn
- PyYAML
- PyTorch Geometric

Details please see requirements.txt

## Downstream Task
PPI task

download the dataset from this Github Releases, put data in the ./data directory. Then,

Running the code:

```
python
from dmt_ppi import DMT_PPI
dmt = DMT_PPI()
#train and val
dmt.fit()
#test
dmt.transform()
```

## Motivation:
The geometric consistency of AlphaFold2 is maintained on the high-dimensional edge representations.

Geoformer could optimize high-dimensional representations to capture complex interaction patterns among amino acids while still maintaining geometric consistency in Euclidean space. In each Geoformer layer, these embeddings are updated iteratively to refine the geometric inconsistency: each node embedding was updated with related pairwise embeddings, and each pairwise embedding was updated by triangular consistency of pairwise embeddings.

However, most protein methods learned protein representations are usually not constrained, leading to performance degradation due to data scarcity, task adaptation, etc. Can we design a loss to satisfy the demand for geometric consistency?



## Codes
for 
[Deep Manifold Graph Auto-Encoder For Attributed Graph Embedding](https://ieeexplore.ieee.org/abstract/document/10095904) 
and
[Deep Manifold Transformation for Protein Representation Learning](https://arxiv.org/abs/2402.09416) 

Codes are based on DLME and KeAP. 

KeAP: [paper](https://openreview.net/forum?id=VbCMhg7MRmj) and [codes](https://github.com/RL4M/KeAP) 

DLME: [paper](https://arxiv.org/abs/2207.03160) and [codes](https://github.com/zangzelin/code_ECCV2022_DLME) 


## Cite the paper

```
@inproceedings{hu2023deep,
  title={Deep manifold graph auto-encoder for attributed graph embedding},
  author={Hu, Bozhen and Zang, Zelin and Xia, Jun and Wu, Lirong and Tan, Cheng and Li, Stan Z},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}

@inproceedings{hu2024deep,
  title={Deep Manifold Transformation for Protein Representation Learning},
  author={Hu, Bozhen and Zang, Zelin and Tan, Cheng and Li, Stan Z},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1801--1805},
  year={2024},
  organization={IEEE}
}
```


## License

DMT is released under the MIT license.
