from utils import *
import os
import cv2
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input', type=str, default='train_images/Image1.jpg', help = 'The input image')
parser.add_argument('--output', type=str, default='out.csv', help = 'The output file')
parser.add_argument('--parallel', action = 'store_true' , help =  'is set, uses the parallel method')
parser.add_argument('--n_steps', type = int , default= 5001 , help = 'Sets the maximum number of training steps')


args = parser.parse_args()

name = args.input
image= cv2.imread(name)
bare_name = 'Img ' + name.split('/')[-1].split('.')[0]

print(args.parallel)

if  os.path.split(args.output)[0] != '':
    os.makedirs(os.path.split(args.output)[0] , exist_ok = True)


add = os.path.split(args.output)

resol = [128]

grids = [1]*6 + [2]*6 + [4]*6 + [8]*6 + [16]*6 + [32]*5
layer = [3]
hidd = [16, 32, 64, 128, 256, 512]*5 + [16, 32, 64, 128, 256]


img_name = add + f'Grid{grids[0]}_w{hidd[0]}_d{layer[0]}_res{resol[0]}_'
 
n_c = 8

gc.collect()
torch.cuda.empty_cache()

output = []
for hid, grid in tqdm(zip(hidd, grids)):
    mid = []
    for layers in layer:
      for res in tqdm(resol):
        for i in range(5):
            print('\nResolution: ', res, ', Number of Grids:', grid, ', Layer:', layers, ', hidden:',hid)
            out = {}


            
            if grid > 2:
                parallel = args.parallel
            else:
                parallel = False


            gc.collect()
            torch.cuda.empty_cache()
            result = train_GPSiren(img_name, image=image, image_sidelength=[res, res], hidden_features = hid,
                                   hidden_layers=layers , grid_ratio=grid, total_steps= args.n_steps, steps_til_summary=5001,
                                   n_batches= n_c , plot=True, show=True, parallel_model = parallel)

            macs, params, flops = flops_counter(hidden_features = hid , hidden_layers = layers, image_size = res, n_grids= grid)

            out['grid'] = grid
            out['resolution'] = res
            out['hidden'] = hid
            out['PSNR'] = 20*np.log10(1.0/np.sqrt(np.min(result['losses'])))
            out['Flops'] = macs
            out['LOG(FLP)'] = flops
            out['params'] = params
            out['Layer'] = layers
            out['Min_Loss'] = np.min(result['losses'])
            out['Step']= n_c

            gc.collect()
            torch.cuda.empty_cache()
            output.append(out)
            mid.append(out)
            print(out)

    df2 = pd.DataFrame(output)
    df2.to_csv(args.output , index=False)

