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
picture: 256, hidden: 96, Grid_ratio:2, Grid Total loss 0.0020818, Total Trainable Params: 29388
![Original Output](./images/grid_result4.jpg?raw=true "Original Model")
picture: 256, hidden: 80, Grid_ratio:2, Grid Total loss 0.0039129, Total Trainable Params: 20652
![Original Output](./images/grid_result5.jpg?raw=true "Original Model")
picture: 256, hidden: 64, Grid_ratio:2, Grid Total loss 0.0074959, Total Trainable Params: 13452
![Original Output](./images/grid_result6.jpg?raw=true "Original Model")
picture: 256, hidden: 56, Grid_ratio:2, Grid Total loss 0.0111716, Total Trainable Params: 10428
![Original Output](./images/grid_result7.jpg?raw=true "Original Model")
picture: 256, hidden: 48, Grid_ratio:2, Grid Total loss 0.0170969, Total Trainable Params: 7788
![Original Output](./images/grid_result8.jpg?raw=true "Original Model")
picture: 256, hidden: 40, Grid_ratio:2, Grid Total loss 0.0265256, Total Trainable Params: 5532
![Original Output](./images/grid_result9.jpg?raw=true "Original Model")
picture: 256, hidden: 32, Grid_ratio:2, Grid Total loss 0.0471402, Total Trainable Params: 3660
![Original Output](./images/grid_result10.jpg?raw=true "Original Model")

picture: 256, hidden: 128, Grid_ratio:4, Grid Total loss 0.0031913, Total Trainable Params: 53808
![Original Output](./images/grid_result2.jpg?raw=true "Original Model")
picture: 256, hidden: 64, Grid_ratio:4, Grid Total loss 0.0471402, Total Trainable Params: 14640
![Original Output](./images/grid_result11.jpg?raw=true "Original Model")

picture: 256, hidden: 128, Grid_ratio:8, Grid Total loss 0.0350236, Total Trainable Params: 58560
![Original Output](./images/grid_result3.jpg?raw=true "Original Model")


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


