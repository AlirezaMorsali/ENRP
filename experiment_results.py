from utils import *
import os
import cv2
from argparse import ArgumentParser
import pandas as pd


parser = ArgumentParser()
parser.add_argument('--input', type=str, default='train_images/Image.jpg', help = 'The input image')
parser.add_argument('--output', type=str, default='out.csv', help = 'The output file')
parser.add_argument('--parallel', default=True, help =  'is set, uses the parallel method')
parser.add_argument('--n_steps', type = int , default= 5001 , help = 'Sets the maximum number of training steps')
parser.add_argument('--batch_size', type = int , default= 16 , help = 'Batch size')

args = parser.parse_args()

name = args.input
image= cv2.imread(name)

print(args.parallel)

if  os.path.split(args.output)[0] != '':
    os.makedirs(os.path.split(args.output)[0] , exist_ok = True)


add = os.path.split(args.output)[0]


resolution = [128]
grids = [1]*6 + [2]*6 + [4]*6 + [8]*6 + [16]*6 + [32]*5
layers = [3]
hidden = [16, 32, 64, 128, 256, 512]*5 + [16, 32, 64, 128, 256]

# Number of training repetitions 
repeat = 10


output = []
for hid, grid in tqdm(zip(hidden, grids)):
    mid = []
    for layer in layers:
      for res in tqdm(resolution):
        for i in range(repeat):
            print('\nResolution: ', res, ', Number of Grids:', grid, ', Layer:', layer, ', hidden:',hid)
            out = {}

            if grid > 2:
                parallel = args.parallel
            else:
                parallel = False

            result = train_GPSiren(image=image, image_sidelength=[res, res], hidden_features = hid,
                                   hidden_layers=layer, grid_ratio=grid, total_steps=args.n_steps, summary_plot=False, 
                                   steps_til_summary=100, store_images=False, n_batches= args.batch_size , parallel_model = parallel, save_model=False)  

            flops, log_flops, params = flops_counter(hidden_features = hid , hidden_layers = layer,
                                                     image_size = res, grid_size=grid)
            out['grid'] = grid
            out['resolution'] = res
            out['hidden'] = hid
            out['PSNR'] = 20*np.log10(1.0/np.sqrt(np.min(result['losses'])))
            out['FLOPs'] = flops
            out['LOG(FLOPs)'] = log_flops
            out['params'] = params
            out['Layer'] = layer
            out['Min_Loss'] = np.min(result['losses'])
            out['Step']= args.batch_size

            output.append(out)
            mid.append(out)
            print(out)

    df = pd.DataFrame(output)
    df.to_csv(args.output , index=False)

