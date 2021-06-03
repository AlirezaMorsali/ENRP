def get_merged_image(sidelength):
    img1 = Image.fromarray(skimage.data.astronaut())
    img2 = Image.fromarray(skimage.data.coffee())
    img3 = Image.fromarray(skimage.data.chelsea())
    img4 = Image.fromarray(skimage.data.colorwheel())

    transform = Compose([
        Resize(size=[sidelength, sidelength]),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])

    img1 = transform(img1)
    img2 = transform(img2)
    img3 = transform(img3)
    img4 = transform(img4)
    img = torch.cat((torch.cat((img1, img2), dim=1), torch.cat((img3, img4), dim=1)), dim=2)
    return img

