# Ensemble Neural Representation Networks

Official implementation of <a href="https://arxiv.org/abs/2110.04124">Ensemble Neural Representation Network</a> 
in pytorch. 

## Abstract
Implicit Neural Representation (INR) has recently attracted considerable attention for storing various types of signals in continuous forms. The existing INR networks require lengthy training processes and high-performance computational resources. In this paper, we propose a novel sub-optimal ensemble architecture for INR that resolves the aforementioned problems. In this architecture, the representation task is divided into several sub-tasks done by independent sub-networks. We show that the performance of the proposed ensemble INR architecture may decrease if the dimensions of sub-networks increase. Hence, it is vital to suggest an optimization algorithm to find the sub-optimal structure of the ensemble network, which is done in this paper. According to the simulation results, the proposed architecture not only has significantly fewer floating-point operations (FLOPs) and less training time, but it also has better performance in terms of Peak Signal to Noise Ratio (PSNR) compared to those of its counterparts.

## Outputs

<!-- ![divergence](https://user-images.githubusercontent.com/44018277/138133928-9ca86652-3bfd-4697-a6f1-59bc7340fa5c.jpg) -->
![divergence](https://user-images.githubusercontent.com/44018277/138134131-a5acd014-6397-43ab-8366-c2c8829aa509.jpg)


## Training
To run the program, you first need to clone the repo and install the requirements using the code below:

```
$ git clone https://github.com/AlirezaMorsali/ENRP.git
$ cd ENRP
$ pip install -r requirements.txt
```
To train the ENRP on your image, run the code below:

```
$python main.py --input [your image] --output[the address to the out file]
```
if you want to train the models in the proposed parallel mode, then pass the argument ```--parallel``` to the above code.

### More descriptions will be added soon!!!
