
def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.astronaut())  
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img
class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        # img = get_merged_image(int(sidelength/2))
        self.pixels = img.permute(1, 2, 0).reshape(-1,3)
        self.coords = get_mgrid(sidelength, 2)
        print(f'pixels shape: {self.pixels.shape}, coords shape: {self.coords.shape}')

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels