import cv2
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.init as init
import time
from tqdm.notebook import tqdm
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import math
from ptflops import get_model_complexity_info
import os
import cv2

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    
class GridDataset(Dataset):
    def __init__(self, image, sidelength=[256, 256], grid_ratio=1, n_batches=1):
        super().__init__()
        self.n_batches = n_batches
        image = self.preprocessImage(image, sidelength)
        self.pixels = self.gridImage(image, grid_ratio)
        self.coords = self.getMgrid(sidelength, grid_ratio)
        
    def preprocessImage(self, image, sidelength):
        image = Image.fromarray(image)
        transform = Compose([
                             Resize(sidelength),
                             ToTensor(),
                             Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
        
        image = transform(image)
        image = image.permute(1, 2, 0)
        return image
    
    def gridImage(self, image, grid_ratio):
        sidelength2 = list(image.size())[1]
        depth = list(image.size())[-1]
        step2 = int(sidelength2/grid_ratio)

        gridImage = None
        gridImages = []
        for i in range(grid_ratio):
            new_image = image[:, i*step2:(i+1)*step2].reshape((grid_ratio, -1, depth)).permute((1,0,2))
            gridImages.append(new_image)
        grid_image = torch.cat(gridImages, dim=1)
        grid_image = torch.unsqueeze(grid_image, dim=len(list(grid_image.size())))
        return grid_image
    
    def getMgrid(self, sidelength, grid_ratio=1, dim=2):
        gridlength1 = int(sidelength[0]/grid_ratio)
        gridlength2 = int(sidelength[1]/grid_ratio)
        tensors = tuple([torch.linspace(-1, 1, steps=gridlength1), torch.linspace(-1, 1, steps=gridlength2)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, 1, dim).repeat(1, grid_ratio**dim, 1)
        mgrid = torch.unsqueeze(mgrid, dim=len(list(mgrid.size())))
        return mgrid
    
    def __len__(self):
        return list(self.pixels.size())[0]

    def __getitem__(self, idx):
        if idx > self.n_batches: raise IndexError
        batch_size = int(list(self.pixels.size())[0]/self.n_batches)
        st = int(idx*batch_size)
        en = int((idx+1)*batch_size)
        if en > list(self.pixels.size())[0]:
            en = list(self.pixels.size())[0]
        return self.coords[st:en], self.pixels[st:en]

    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.0):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output, coords


# Grid Non-parallel Siren
class GSiren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, grid_ratio, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.0):
        super(GSiren, self).__init__()
        self.n_grids = grid_ratio**in_features
        
        self.net = nn.ModuleList([])
        for i in range(self.n_grids):
            self.net.append(Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear, first_omega_0, hidden_omega_0))

    def forward(self, coords, i=None, j=None):
        output = None
        tmpI = None
        for i in range(self.n_grids):
            outputt, tmp = self.net[i](coords[:, i, :, :].squeeze())
            outputt = outputt.unsqueeze(dim=1)
            outputt = outputt.unsqueeze(dim=len(list(outputt.size())))
            if i==0:
                tmpI = outputt
            else:
                tmpI = torch.cat((tmpI, outputt), dim=1)
        output = tmpI
        return output, coords


class GridLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, grid_ratio=1, bias=True, device=None, dtype=None):
        super(GridLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_ratio = grid_ratio
        self.weight = torch.nn.Parameter(torch.empty((grid_ratio, out_features, in_features)))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(grid_ratio, out_features, 1))
        else:
            self.bias = None
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(self.weight, input)
        if not self.bias is None:
            output = output + self.bias
        return output


class GPSineLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_ratio, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = GridLinear(in_features, out_features, grid_ratio, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
                    
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


# Grid Parallel Siren
class GPSiren(nn.Module):
    def __init__(self, in_features, grid_hidden_features, hidden_layers, grid_ratio, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.0):
        super().__init__()

        self.in_features = in_features
        self.grid_hidden_features = grid_hidden_features
        self.grid_ratio = grid_ratio
        self.n_grids = grid_ratio**in_features
        self.net = []
        self.net.append(GPSineLayer(in_features, grid_hidden_features, grid_ratio=self.n_grids, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(GPSineLayer(grid_hidden_features, grid_hidden_features, grid_ratio=self.n_grids, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = GridLinear(grid_hidden_features, out_features, self.n_grids)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / grid_hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / grid_hidden_features) / hidden_omega_0)
            
            self.net.append(final_linear)
        else:
            self.net.append(GPSineLayer(grid_hidden_features, out_features, grid_ratio=self.n_grids, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output, coords

    
def renderGridImage(gridImage, image_size=[256, 256]):
    grid_ratio = int(math.sqrt(list(gridImage.size())[1]))
    
    tmpJs = []
    for i in range(grid_ratio):
        tmpIs = []
        for j in range(grid_ratio):
            cur = i*grid_ratio + j
            tmpIs.append(gridImage[:, cur, :, :].reshape(int(image_size[0]/grid_ratio), int(image_size[1]/grid_ratio), 3))
        tmpJs.append(torch.vstack(tmpIs))
    model_out = torch.hstack(tmpJs)

    out_f = model_out.cpu().detach().numpy()
    
    model_output = 0.5 + 0.5 * out_f
    model_output = (model_output - model_output.min())
    model_output = (model_output / model_output.max())

    return model_output[...,::-1]



def train_GPSiren(image, image_sidelength=[256, 256], in_features=2, out_features=3, grid_ratio=2, 
                    hidden_layers=3, hidden_features=32, total_steps=500,
                    summary_plot=False, steps_til_summary=100, store_images=False, save_frame_rate = 5,
                    cuda=True, parallel_model=True, n_batches=1, save_model = False, checkpoints_dir = './output_model'):

    if parallel_model:
        img_siren = GPSiren(in_features=in_features, grid_hidden_features=hidden_features, grid_ratio=grid_ratio, out_features = out_features, 
                        hidden_layers=hidden_layers, outermost_linear=True, first_omega_0=30.0, hidden_omega_0=30.0)
    else:
        img_siren = GSiren(in_features=in_features, out_features=out_features, hidden_features=hidden_features, grid_ratio=grid_ratio, 
                        hidden_layers=hidden_layers, outermost_linear=True, first_omega_0=30.0, hidden_omega_0=30.0)

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    dataloader = GridDataset(image, sidelength=image_sidelength, grid_ratio=grid_ratio, n_batches=n_batches)

    if cuda:
        img_siren.cuda()

    losses = []
    output_images = []
    
    totalTime = 0
    bar = tqdm(range(total_steps),leave=False)
    minLoss = -1
    minLossStep = -1
    for step in bar:
        if minLossStep != -1 and step - minLossStep > 200:
            break
        t1 = time.time()
        optim.zero_grad()
        totalLoss = 0
        model_outputs = []
        for batch in range(n_batches):
            model_input, ground_truth = dataloader[batch]
            if cuda:
                model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
            
            model_output, coords = img_siren(model_input)
            loss = 0
            n_pixels = list(ground_truth.size())[0]
            n_grids = list(model_output.size())[1]
            maxP = 2**18
            loss_step = max(1, int(maxP/n_grids))
            n_step = int(n_pixels/loss_step)
            st_step = 0
            en_step = 0
            for i in range(n_step):
                st_step = i*loss_step
                en_step = (i+1)*loss_step
                loss = loss + ((en_step - st_step) * ((model_output[st_step:en_step] - ground_truth[st_step:en_step])**2).mean())
            if n_pixels > en_step:
                loss = loss + ((n_pixels - en_step) * ((model_output[en_step:] - ground_truth[en_step:])**2).mean())
            loss.backward()
            totalLoss = totalLoss + loss
            model_outputs.append(model_output)
        optim.step()
        totalLoss = totalLoss / len(dataloader)
        totalTime += (time.time() - t1)

        if summary_plot:
            if step % steps_til_summary ==0:
                print("Step %d, Total loss %0.6f Total Time %0.6f" % (step, totalLoss, totalTime))

        if store_images and (step % save_frame_rate == 0):
            out_f = renderGridImage(torch.cat(model_outputs, dim=0), image_size=image_sidelength)
            output_images.append(out_f)
        
        losses.append(totalLoss.cpu().detach().numpy().item())
        if minLoss == -1 or minLoss > min(losses):
            minLoss = min(losses)
            minLossStep = step
        
    if save_model:
        os.makedirs(checkpoints_dir, exist_ok=True)
        torch.save(img_siren.state_dict(),
        os.path.join(checkpoints_dir, 'model_final_G{grid_ratio}_L{hidden_layers}_H{hidden_features}.pth'))
        
        # Save output img
        cv2.imwrite(os.path.join(checkpoints_dir, 'img_G{grid_ratio}_L{hidden_layers}_H{hidden_features}.png'),out_f)

    return {'losses':losses, 'time':totalTime, 'images': output_images}


# FLOPs Counter
# From https://github.com/sovrasov/flops-counter.pytorch
def flops_to_string(flops, units=None, precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


def flops_counter(hidden_features, hidden_layers, image_size, grid_size):
    img_siren = Siren(in_features=2, out_features=3, hidden_features=hidden_features, 
                      hidden_layers=hidden_layers, outermost_linear=True)
    
    macs, params = get_model_complexity_info(img_siren, (1, image_size**2, 2), as_strings=False,
                                              print_per_layer_stat=False, verbose=False)
    
    log_flops  = np.log10(macs*2)
    flops = flops_to_string(macs*2)
    params = params_to_string(params * grid_size ** 2)
    return flops, log_flops, params
