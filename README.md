# Ensemble Neural Representation Networks
[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlirezaMorsali/ENRP/blob/main/ENRP_Demo.ipynb)<br>

[Milad Soltany Kadarvish](https://miladsoltany.github.io/)\*,
[Hesam Mojtahedi](https://scholar.google.com/citations?user=Kr2UwU0AAAAJ&hl=en/)\*,
[Hossein Entezari Zarch](https://scholar.google.com/citations?user=xhVKvhIAAAAJ&hl=en/)\*,
[Amirhossein Kazerouni](https://amirhossein-kz.github.io/)\*,
[Alireza Morsali](https://scholar.google.com/citations?user=y-RVrUkAAAAJ&hl=en/),<br>
[Azra Abtahi](https://scholar.google.com/citations?user=5UdXGpYAAAAJ&hl=en),
[Farokh Marvasti](http://acri.sharif.ir/resume/marvasti/)

\* Equal contribution

This is the official implementation of <a href="https://arxiv.org/abs/2110.04124">Ensemble Neural Representation Network</a> in pytorch. 

![Algorithm](https://user-images.githubusercontent.com/61879630/138571726-c257fa71-4994-43f4-8a77-0e0ec67b70b1.png)

## Abstract
Implicit Neural Representation (INR) has recently attracted considerable attention for storing various types of signals in continuous forms. The existing INR networks require lengthy training processes and high-performance computational resources. In this paper, we propose a novel sub-optimal ensemble architecture for INR that resolves the aforementioned problems. In this architecture, the representation task is divided into several sub-tasks done by independent sub-networks. We show that the performance of the proposed ensemble INR architecture may decrease if the dimensions of sub-networks increase. Hence, it is vital to suggest an optimization algorithm to find the sub-optimal structure of the ensemble network, which is done in this paper. According to the simulation results, the proposed architecture not only has significantly fewer floating-point operations (FLOPs) and less training time, but it also has better performance in terms of Peak Signal to Noise Ratio (PSNR) compared to those of its counterparts.

## Outputs
https://user-images.githubusercontent.com/61879630/138573602-7d80571e-ec47-4d13-8cfc-8ba881c137db.mp4

Results for 500 training steps.
<!-- ![divergence](https://user-images.githubusercontent.com/44018277/138134131-a5acd014-6397-43ab-8366-c2c8829aa509.jpg)
 -->
## Training
To run the program, you first need to clone the repo and install the requirements using the code below:

```
$ git clone https://github.com/AlirezaMorsali/ENRP.git
$ cd ENRP
$ pip install -r requirements.txt
```
To train the ENRP with your configuration, run the code below:

```
$ python single_model.py --input [your image] --grid[grid size] --depth[number of hidden layers] --width[number of hidden features] --n_steps 501 --batch_size 8
```

Grid size should be of two powers `(1,2,4,8,16,32,...)`.

## Citation
If you find our work useful in your research, please cite:

```
@article{kadarvish2021ensemble,
  title={Ensemble Neural Representation Networks},
  author={Kadarvish, Milad Soltany and Mojtahedi, Hesam and Zarch, Hossein Entezari and Kazerouni, Amirhossein and Morsali, Alireza and Abtahi, Azra and Marvasti, Farokh},
  journal={arXiv preprint arXiv:2110.04124},
  year={2021}
}
```
