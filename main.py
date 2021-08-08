import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import math

import time


class GridDataset(Dataset):
    def __init__(self, image, sidelength=256, grid_ratio=1):
        super().__init__()
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
        print(image.size())
        image = image.permute(1, 2, 0)
        return image
    
    def gridImage(self, image, grid_ratio):
        sidelength = list(image.size())[0]
        depth = list(image.size())[-1]
        step = int(sidelength/grid_ratio)

        gridImage = None
        for i in range(grid_ratio):
            for j in range(grid_ratio):
                if i==0 and j==0:
                    grid_image = image[i*step:(i+1)*step, j*step:(j+1)*step].reshape((-1, 1, depth))
                else:
                    grid_image = torch.cat((grid_image, image[i*step:(i+1)*step, j*step:(j+1)*step].reshape((-1, 1, depth))), dim=1)
        grid_image = torch.unsqueeze(grid_image, dim=len(list(grid_image.size())))
        return grid_image
    
    def getMgrid(self, sidelength, grid_ratio=1, dim=2):
        gridlength = int(sidelength/grid_ratio)
        tensors = tuple(dim * [torch.linspace(-1, 1, steps=gridlength)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, 1, dim).repeat(1, grid_ratio**dim, 1)
        mgrid = torch.unsqueeze(mgrid, dim=len(list(mgrid.size())))
        return mgrid
    
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords, self.pixels


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

class GSiren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, grid_ratio, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.0):
        super(GSiren, self).__init__()
        self.grid_ratio = grid_ratio
        
        self.net = nn.ModuleList([])
        for i in range(grid_ratio):
          for j in range(grid_ratio):
            self.net.append(Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear, first_omega_0, hidden_omega_0))

    def forward(self, coords, i=None, j=None):
        step = int(list(coords.size())[0]/(self.grid_ratio**2))
        output = None
        coords_out = None
        tmpI = None
        for i in range(self.grid_ratio):
          for j in range(self.grid_ratio):
            cur = i*self.grid_ratio+j
            outputt, tmp = self.net[cur](coords[:, cur, :, :].squeeze())
            outputt = outputt.unsqueeze(dim=1)
            outputt = outputt.unsqueeze(dim=len(list(outputt.size())))
            if i==0 and j==0:
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



def renderGridImage(gridImage):
    tmpI = None
    tmpJ = None
    grid_ratio = int(math.sqrt(list(gridImage.size())[1]))
    grid_size = int(math.sqrt(list(gridImage.size())[0]))
    for i in range(grid_ratio):
        for j in range(grid_ratio):
            cur = i*grid_ratio + j
            if j==0:
                tmpJ = gridImage[:, cur, :, :].reshape(grid_size, grid_size, 3)
            else:
                tmpJ = torch.hstack((tmpJ, gridImage[:, cur, :, :].reshape(grid_size, grid_size, 3)))
        if i==0:
            tmpI = tmpJ
        else:
            tmpI = torch.vstack((tmpI, tmpJ))
    model_output = tmpI
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    axes[0].imshow(model_output.cpu().detach().numpy()*0.5+0.5)
    plt.show()


def train_GPSiren(image, image_sidelength, in_features=2, hidden_features=32, grid_ratio=2, out_features=3, total_steps=500, step_til_summary=100, plot=False):
    total_steps = 501
    steps_til_summary = 100

    img_siren = GPSiren(in_features=in_features, grid_hidden_features=hidden_features, grid_ratio=grid_ratio, out_features = out_features, 
                    hidden_layers=3, outermost_linear=True, first_omega_0=30.0, hidden_omega_0=30.0)

    # img_siren = GSiren(in_features=in_features, out_features=out_features, hidden_features=hidden_features, grid_ratio=grid_ratio, 
    #                 hidden_layers=3, outermost_linear=True, first_omega_0=30.0, hidden_omega_0=30.0)

    img_siren.cuda()
    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
    n_params = count_parameters(img_siren, shtable=False)

    dataloader = GridDataset(image, sidelength=image_sidelength, grid_ratio=grid_ratio)
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    losses = []
    totalTime = 0
    for step in range(total_steps):
        t1 = time.time()
        model_output, coords = img_siren(model_input)

        loss = 0
        n_grids = list(ground_truth.size())[1]
        for i in range(n_grids):
            loss = loss + ((model_output[:, i, :, :] - ground_truth[:, i, :, :])**2).mean()/n_grids

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        totalTime += (time.time() - t1)
        if step_til_summary != 0 and not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            if plot:
                renderGridImage(model_output)
        losses.append(loss.cpu().detach().numpy())

    return {'losses':losses, 'n_params':n_params, 'time':totalTime}

image = skimage.data.astronaut()
result = train_GPSiren(image=image, image_sidelength=256, hidden_features=16, grid_ratio=16)