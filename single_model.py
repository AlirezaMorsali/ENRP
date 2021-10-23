from utils import *
import cv2
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--input', type=str, default='train_images/Image.jpg', help = 'The input image')

parser.add_argument('--grid', type=int, default=32, help = 'input grid size')
parser.add_argument('--depth', type=int, default=2, help = 'hidden layers')
parser.add_argument('--width', type=int, default=64, help = 'hidden features')
parser.add_argument('--sidelength', type=int, default=128, help = 'image sidelength')

parser.add_argument('--parallel', default=True, help =  'is set, uses the parallel method')
parser.add_argument('--n_steps', type = int , default= 500 , help = 'Sets the maximum number of training steps')
parser.add_argument('--batch_size', type = int , default= 8 , help = 'Batch size')
parser.add_argument('--save_model', default=True, help="saves output model")


args = parser.parse_args()

name = args.input
image= cv2.imread(name)

print(args.parallel)


if args.grid > 2:
    parallel = args.parallel
else:
    parallel = False

out = {}
result = train_GPSiren(image=image, image_sidelength=[args.sidelength, args.sidelength], hidden_features = args.width,
                        hidden_layers=args.depth, grid_ratio=args.grid, total_steps=args.n_steps, summary_plot=False,
                        steps_til_summary=100, n_batches= args.batch_size , parallel_model = parallel, save_model=args.save_model)  
  

flops, log_flops, params = flops_counter(hidden_features = args.width , hidden_layers = args.depth,
                                            image_size = args.sidelength, grid_size=args.grid)
out['grid'] = args.grid
out['resolution'] = args.sidelength
out['width'] = args.width
out['PSNR'] = 20*np.log10(1.0/np.sqrt(np.min(result['losses'])))
out['Flops'] = flops
out['LOG(FLOPs)'] = log_flops
out['params'] = params
out['depth'] = args.depth
out['Min_Loss'] = np.min(result['losses'])
out['n_batches']= args.batch_size

print(out)
