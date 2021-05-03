# EffNeurep
## The Image:
![Original Output](./images/image.jpg?raw=true "Original Model")
## Original Model Output with Different Hidden Sizes:
picture: 256, hidden: 512, Total loss 0.000017, Total Trainable Params: 791043
![Original Output](./images/original_output(0).jpg?raw=true "Original Model")
picture: 256, hidden: 256, Total loss 0.000082, Total Trainable Params: 198915
![Original Output](./images/original_output.jpg?raw=true "Original Model")
picture: 256, hidden: 128, Total loss 0.000828, Total Trainable Params: 50307
![Original Output](./images/original_output(1).jpg?raw=true "Original Model")
picture: 256, hidden: 64, Total loss 0.006961, Total Trainable Params: 12867
![Original Output](./images/original_output(2).jpg?raw=true "Original Model")
picture: 256, hidden: 32, Total loss 0.039233, Total Trainable Params: 3363
![Original Output](./images/original_output(3).jpg?raw=true "Original Model")

## Step 1: Gridding the Siren
picture: 256, hidden: 128, Grid_ratio:2, Grid Total loss 0.0007812, Total Trainable Params: 51468
![Original Output](./images/grid_result1.jpg?raw=true "Original Model")
picture: 256, hidden: 96, Grid_ratio:2, Grid Total loss 0.000571, Total Trainable Params: 29388
![Original Output](./images/grid_result4.jpg?raw=true "Original Model")
picture: 256, hidden: 80, Grid_ratio:2, Grid Total loss 0.001055, Total Trainable Params: 20652
![Original Output](./images/grid_result5.jpg?raw=true "Original Model")
picture: 256, hidden: 64, Grid_ratio:2, Grid Total loss 0.001914, Total Trainable Params: 13452
![Original Output](./images/grid_result6.jpg?raw=true "Original Model")
picture: 256, hidden: 56, Grid_ratio:2, Grid Total loss 0.002929, Total Trainable Params: 10428
![Original Output](./images/grid_result7.jpg?raw=true "Original Model")
picture: 256, hidden: 48, Grid_ratio:2, Grid Total loss 0.004338, Total Trainable Params: 7788
![Original Output](./images/grid_result8.jpg?raw=true "Original Model")
picture: 256, hidden: 40, Grid_ratio:2, Grid Total loss 0.006983, Total Trainable Params: 5532
![Original Output](./images/grid_result9.jpg?raw=true "Original Model")
picture: 256, hidden: 32, Grid_ratio:2, Grid Total loss 0.011597, Total Trainable Params: 3660
![Original Output](./images/grid_result10.jpg?raw=true "Original Model")

picture: 256, hidden: 128, Grid_ratio:4, Grid Total loss 0.0002337, Total Trainable Params: 53808
![Original Output](./images/grid_result2.jpg?raw=true "Original Model")
picture: 256, hidden: 64, Grid_ratio:4, Grid Total loss 0.003049, Total Trainable Params: 14640
![Original Output](./images/grid_result11.jpg?raw=true "Original Model")

picture: 256, hidden: 128, Grid_ratio:8, Grid Total loss 0.000810, Total Trainable Params: 58560
![Original Output](./images/grid_result3.jpg?raw=true "Original Model")

### Improve Tiny Siren Nets:
#### change of omegas:
picture: 256, hidden: 32, Total loss 0.039233, Total Trainable Params: 3363
![Original Output](./images/original_output(3).jpg?raw=true "Original Model")

picture: 256, hidden: 32, outermost_linear: False, Total loss 0.051598, Total Trainable Params: 3363
![Original Output](./images/tiny_output.jpg?raw=true "Original Model")
picture: 256, hidden: 32, first_omega_0:60, hidden_omega:30, Total loss 0.055761, Total Trainable Params: 3363
![Original Output](./images/tiny_output3.jpg?raw=true "Original Model")
picture: 256, hidden: 32, first_omega_0:30, hidden_omega:60, Total loss 0.041092, Total Trainable Params: 3363
![Original Output](./images/tiny_output4.jpg?raw=true "Original Model")
picture: 256, hidden: 32, first_omega_0:30, hidden_omega:120, Total loss 0.040188, Total Trainable Params: 3363
![Original Output](./images/tiny_output5.jpg?raw=true "Original Model")
picture: 256, hidden: 32, first_omega_0:30, hidden_omega:240, Total loss 0.043150, Total Trainable Params: 3363
![Original Output](./images/tiny_output6.jpg?raw=true "Original Model")
picture: 256, hidden: 32, first_omega_0:20, hidden_omega:120, Total loss 0.042351, Total Trainable Params: 3363
![Original Output](./images/tiny_output7.jpg?raw=true "Original Model")
picture: 256, hidden: 32, first_omega_0:10, hidden_omega:120, Total loss 0.049642, Total Trainable Params: 3363
![Original Output](./images/tiny_output8.jpg?raw=true "Original Model")
#### change of layers:
picture: 256, layers:2, hidden: 40, Total loss 0.040570, Total Trainable Params: 3523
![Original Output](./images/tiny_output9.jpg?raw=true "Original Model")
picture: 256, layers:1, hidden: 56, Total loss 0.056993, Total Trainable Params: 3531
![Original Output](./images/tiny_output10.jpg?raw=true "Original Model")
picture: 256, layers:4, hidden: 28, Total loss 0.037657, Total Trainable Params: 3419
![Original Output](./images/tiny_output11.jpg?raw=true "Original Model")
picture: 256, layers:5, hidden: 25, Total loss 0.036021, Total Trainable Params: 3403
![Original Output](./images/tiny_output12.jpg?raw=true "Original Model")
picture: 256, layers:6, hidden: 23, Total loss 0.042324, Total Trainable Params: 3452

![Original Output](./images/tiny_output13.jpg?raw=true "Original Model")
##### change of omegas:
picture: 256, layers:5, hidden:25, first_omega_0:30, hidden_omega:30, Total loss 0.036021, Total Trainable Params: 3403
![Original Output](./images/tiny_output12.jpg?raw=true "Original Model")
picture: 256, layers:5, hidden:25, first_omega_0:60, hidden_omega:30, Total loss 0.084523, Total Trainable Params: 3403
![Original Output](./images/tiny_output14.jpg?raw=true "Original Model")
picture: 256, layers:5, hidden:25, first_omega_0:30, hidden_omega:60, Total loss 0.033875, Total Trainable Params: 3403
![Original Output](./images/tiny_output15.jpg?raw=true "Original Model")
picture: 256, layers:5, hidden:25, first_omega_0:30, hidden_omega:90, Total loss 0.038705, Total Trainable Params: 3403
![Original Output](./images/tiny_output16.jpg?raw=true "Original Model")
picture: 256, layers:5, hidden:25, first_omega_0:20, hidden_omega:60, Total loss 0.036265, Total Trainable Params: 3403
![Original Output](./images/tiny_output17.jpg?raw=true "Original Model")
picture: 256, layers:5, hidden:25, first_omega_0:10, hidden_omega:60, Total loss 0.034360, Total Trainable Params: 3403
![Original Output](./images/tiny_output18.jpg?raw=true "Original Model")
picture: 256, layers:5, hidden:25, first_omega_0:5, hidden_omega:60, Total loss 0.044639, Total Trainable Params: 3403
![Original Output](./images/tiny_output19.jpg?raw=true "Original Model")
picture: 256, layers:5, hidden:25, first_omega_0:1, hidden_omega:60, Total loss 0.072873, Total Trainable Params: 3403
![Original Output](./images/tiny_output20.jpg?raw=true "Original Model")
picture: 256, hidden: 32, first_omega_0:300, Total loss 0.682163, Total Trainable Params: 3363
![Original Output](./images/tiny_output1.jpg?raw=true "Original Model")
picture: 256, hidden: 32, first_omega_0:300, hidden_omega:300, Total loss 0.957278, Total Trainable Params: 3363
![Original Output](./images/tiny_output2.jpg?raw=true "Original Model")

### Grids with best omegas we found:
picture: 256, gird:2, layers:5, hidden:100, first_omega_0:30, hidden_omega:60, Total loss 0.000187, Total Trainable Params: 52212
![Original Output](./images/grid_result12.jpg?raw=true "Original Model")
picture: 256, gird:2, layers:5, hidden:50, first_omega_0:30, hidden_omega:60, Total loss 0.001837, Total Trainable Params: 13612
![Original Output](./images/grid_result13.jpg?raw=true "Original Model")

### Skip Connection:
picture: 256, hidden: 128, Total loss 0.003690, Total Trainable Params: 50307
![Original Output](./images/skip_model.jpg?raw=true "Original Model")
#### Skip Connection with Weights:
picture: 256, hidden: 256, Total loss 0.000167, Total Trainable Params: 199427
![Original Output](./images/skip_model2.jpg?raw=true "Original Model")
picture: 256, hidden: 128, Total loss 0.001049, Total Trainable Params: 50563
![Original Output](./images/skip_model1.jpg?raw=true "Original Model")
picture: 256, hidden: 64, Total loss 0.007480, Total Trainable Params: 12995
![Original Output](./images/skip_model3.jpg?raw=true "Original Model")
picture: 256, hidden: 32, Total loss 0.037725, Total Trainable Params: 3427
![Original Output](./images/skip_model4.jpg?raw=true "Original Model")

* Image into 4 block, and 4 smaller networks
.. Figuring out if there are any improvement
* Image into 16 block, and 16 smaller networks
.. Figuring out if there are any improvement
* Examining the frequency to see at which point we are having loss when scaling down the network


* How does the original network perform on the blocks of image


### Result:
      What are the pros and cons of the Grid?
      
## Step 2: Improve the shortcomming


# Ideas:




## Diferent Architecture for small parts of an image:
## Initialilzaion
Smaller blocks may be sensitive to initialization
## Resrnet


## Frequency split, parallel networks
I(x,y) = N_lf(x,y) + N_hf(x,y)

## Resedual learning
Train grid networks on the block
Train a samll network on the resedual 

## Last layer as DCT 
Check to see if it has been done in other works
we can do both the discrete or training the continous coeeficients

## Wavelet


