# Mixing Self-Attention and Convolution Network: A Uniﬁed Framework for Multisource Remote Sensing Data Classification
This is the offical code for paper "Mixing Self-Attention and Convolution: A Unified Framework for Multisource Remote Sensing Data Classification", [Paper](https://ieeexplore.ieee.org/document/10236462).

Author: Ke Li; Di Wang; Xu Wang; Gang Liu; Zili Wu; Quan Wang

Key Laboratory of Smart Human–Computer Interaction and Wearable Technology of Shaanxi Province, School of Computer Science and Technology, Xidian University, Xi’an, China

## Abstract
Convolution and self-attention are two powerful techniques for multisource remote sensing (RS) data fusion that have been widely adopted in Earth observation tasks. However, convolutional neural networks (CNNs) are inadequate for fully mining contextual information and representing the sequence attributes of spectral signatures. In addition, the specific self-attention mechanism often comes with high-computational costs, which hinders its application in the field of RS. To overcome the above limitations, this article proposes a unified framework called “mixing self-attention and convolution network (MACN)” for comprehensive feature extraction and efficient feature fusion. First, the proposed MACN utilizes two adaptive CNN encoders (ACEs) to extract shallow convolutional features from multisource RS data. Second, taking the complexity and varying scales of RS data into account, the proposed mixing self-attention and convolution Transformer (MACT) layer achieves local and global multiscale perception through an elegant integration of self-attention and convolution. MACT can extract abundant spatial and high-dimensional information (e.g., spectral and elevation information) while maintaining minimal computational overhead compared with pure convolution or self-attention counterparts. Finally, a multisource cross-guided fusion (MCGF) module is designed to achieve deep fusion of multisource RS data features. MCGF utilizes a carefully designed cross-modal attention mechanism to capture the interaction between multisource data and aggregate contextual information. Extensive tests on six public RS datasets have shown that our method outperforms other multisource fusion models, delivering state-of-the-art (SOTA) results on multiple RS data fusion tasks without specific tuning. 
## Usage
### dataset utilization
Please modify line 20-22 in trentoTrain.py for the dataset details.
### Training
Train the HSI and LiDAR-based DSM   
```
python trentoTrain.py 
```

## Results
All the results are cited from original paper. More details can be found in the paper.

| dataset      | OA     | Kappa  |
| --------     | ------ | ------ |
| MUUFL        | 90.66% | 87.73% |
| Houston2013  | 99.81% | 99.80% |
| Houston2018  | 84.58% | 78.41% |
| Trento       | 99.73% | 99.66% |
| Augsburg     | 97.56% | 96.37% |
| Berlin       | 84.30% | 74.87% |

## Citation
If you found this code useful, please cite the paper. Welcome 👍Fork and Star👍, then I will let you know when we update.

```
@ARTICLE{10236462,
  author={Li, Ke and Wang, Di and Wang, Xu and Liu, Gang and Wu, Zili and Wang, Quan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Mixing Self-Attention and Convolution: A Unified Framework for Multisource Remote Sensing Data Classification}, 
  year={2023},
  volume={61},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2023.3310521}}
```
