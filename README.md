# GMI (Graphical Mutual Information)
Graph Representation Learning via Graphical Mutual Information Maximization (Peng Z, Huang W, Luo M, *et al.*, WWW 2020): [https://arxiv.org/abs/2002.01169](https://arxiv.org/abs/2002.01169)

![image](https://github.com/zpeng27/GMI/blob/master/illustration.jpg)

## Overview
Note that we propose two variants of GMI in the paper, the one is GMI-mean, and the other is GMI-adaptive. Since GMI-mean often outperforms GMI-adaptive (see the experiments in the paper), here we give a PyTorch implementation of GMI-mean. To make GMI more practical, we provide an alternative solution to compute FMI. Such a solution still ensures the effectiveness of GMI and improves the efficiency greatly. The repository is organized as follows:

- `data/` includes three benchmark datasets;
- `models/` contains the implementation of the GMI pipeline (`gmi.py`) and the logistic regression classifier (`logreg.py`);
- `layers/` contains the implementation of a standard GCN layer (`gcn.py`), the bilinear discriminator (`discriminator.py`), and the mean-pooling operator (`avgneighbor.py`);
- `utils/` contains the necessary processing tool (`process.py`).

To better understand the code, we recommend that you could read the code of DGI/Petar (https://arxiv.org/abs/1809.10341) in advance. Besides, you could further optimize the code based on your own needs. We display it in an easy-to-read form.

## Requirements

  * PyTorch 1.2.0
  * Python 3.6

## Usage

```python execute.py```

## Cite
Please cite our paper if you make advantage of GMI in your research:

```
@inproceedings{
peng2020graph,
title="{Graph Representation Learning via Graphical Mutual Information Maximization}",
author={Peng, Zhen and Huang, Wenbing and Luo, Minnan and Zheng, Qinghua and Rong, Yu and Xu, Tingyang and Huang, Junzhou},
booktitle={Proceedings of The Web Conference},
year={2020},
doi={https://doi.org/10.1145/3366423.3380112},
}
```
