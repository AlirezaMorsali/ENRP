def get_brick_tensor(sidelength):
    img = Image.fromarray(skimage.data.brick())  
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img
class ImageFittingBrick(Dataset):
    def __init__(self, sidelength, grid=False):
        super().__init__()
        img = get_brick_tensor(sidelength)
        print(img.shape)
        if not grid:
          self.pixels = img.permute(1, 2, 0).reshape(-1,1)
        else:
          self.pixels = img.permute(1, 2, 0)
        self.coords = get_mgrid(sidelength, 2, grid)
        print(f'pixels shape: {self.pixels.shape}, coords shape: {self.coords.shape}')

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels

def get_grass_tensor(sidelength):
    img = Image.fromarray(skimage.data.grass())  
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img
class ImageFittingGrass(Dataset):
    def __init__(self, sidelength, grid=False):
        super().__init__()
        img = get_grass_tensor(sidelength)
        print(img.shape)
        if not grid:
          self.pixels = img.permute(1, 2, 0).reshape(-1,1)
        else:
          self.pixels = img.permute(1, 2, 0)
        self.coords = get_mgrid(sidelength, 2, grid)
        print(f'pixels shape: {self.pixels.shape}, coords shape: {self.coords.shape}')

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels


cameraman = ImageFittingBrick(256, grid=True)
dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)
torch.autograd.set_detect_anomaly(True)
report2 = train_GSiren()

cameraman = ImageFittingGrass(256, grid=True)
dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)
torch.autograd.set_detect_anomaly(True)
report3 = train_GSiren()