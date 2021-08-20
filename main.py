import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
import os
import time
from prettytable import PrettyTable

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import math


def count_parameters(model, shtable=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    if shtable:
      print(table)
      print(f"Total Trainable Params: {total_params}")
    return total_params

# Pixels -> num of pixels of each grid * num of grids * num of output features * 1
# Coords -> num of pixels of each grid * num of grids * num of input features * 1
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


# Weight -> num of grids * out_features * in_features
# Input -> num of pixels * num of grids * in_features * 1
# Weight * Input -> num of pixels * num of grids * out_features * 1
# Bias -> num of grids * out_features * 1
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
    tmpI = None
    tmpJ = None
    grid_ratio = int(math.sqrt(list(gridImage.size())[1]))
    grid_size = int(math.sqrt(list(gridImage.size())[0]))
    
    tmpJs = []
    for i in range(grid_ratio):
        tmpIs = []
        for j in range(grid_ratio):
            cur = i*grid_ratio + j
            tmpIs.append(gridImage[:, cur, :, :].reshape(int(image_size[0]/grid_ratio), int(image_size[1]/grid_ratio), 3))
        tmpJs.append(torch.vstack(tmpIs))
    model_output = torch.hstack(tmpJs)
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    axes[0].imshow(model_output.cpu().detach().numpy()*0.5+0.5)
    plt.savefig('output.png')
    plt.show()


def train_GPSiren(image, image_sidelength=[256, 256], in_features=2, out_features=3, grid_ratio=2, 
                    hidden_layers=3, hidden_features=32,
                    total_steps=500, steps_til_summary=100, plot=False,
                    cuda=True, parallel_model=True, n_batches=1):

    if parallel_model:
        img_siren = GPSiren(in_features=in_features, grid_hidden_features=hidden_features, grid_ratio=grid_ratio, out_features = out_features, 
                        hidden_layers=3, outermost_linear=True, first_omega_0=30.0, hidden_omega_0=30.0)
    else:
        img_siren = GSiren(in_features=in_features, out_features=out_features, hidden_features=hidden_features, grid_ratio=grid_ratio, 
                        hidden_layers=3, outermost_linear=True, first_omega_0=30.0, hidden_omega_0=30.0)

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
    n_params = count_parameters(img_siren, shtable=False)

    dataloader = GridDataset(image, sidelength=image_sidelength, grid_ratio=grid_ratio, n_batches=n_batches)
    print('dataloader')
    
    if cuda:
        img_siren.cuda()

    losses = []
    totalTime = 0
    for step in range(total_steps):
        t1 = time.time()
        optim.zero_grad()
        model_generated_image = None
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
        if steps_til_summary != 0 and not step % steps_til_summary:
            print("Step %d, Total loss %0.6f Total Time %0.6f" % (step, totalLoss, totalTime))
            if plot:
                renderGridImage(torch.cat(model_outputs, dim=0), image_size=image_sidelength)
        losses.append(totalLoss.cpu().detach().numpy().item())
    print(f'time: {totalTime}')
    return {'losses':losses, 'n_params':n_params, 'time':totalTime}

# image = skimage.data.astronaut()
# result = train_GPSiren(image=image, image_sidelength=[256, 256], hidden_features=32, grid_ratio=4, total_steps=501, step_til_summary=100, plot=True)