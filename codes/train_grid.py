def train_grid(hf=256, grid_ratio=2, hl=3, fo=30.0, ho=30.0):
    datas = []
    pic_step = int(mISLen/grid_ratio)
    for i in range(grid_ratio):
        for j in range(grid_ratio):
            DataLoader(datas.append(ImageFitting2(mainImage[:, i*pic_step:(i+1)*pic_step, j*pic_step:(j+1)*pic_step], pic_step)), batch_size=1, pin_memory=True, num_workers=0)
    
    sirens = []
    for i in range(grid_ratio):
        for j in range(grid_ratio):
            sirens.append(Siren(in_features=2, out_features=3, hidden_features=int(hf/grid_ratio), 
                    hidden_layers=hl, outermost_linear=True, first_omega_0=fo, hidden_omega_0=ho).cuda())
    tmp_par = count_parameters(sirens[0])
    nparams_model = grid_ratio*grid_ratio*tmp_par
    print(f'>>>>>>>> Total Trainable Params: {grid_ratio*grid_ratio*tmp_par}')
    output_image = np.zeros((mISLen, mISLen, 3))
    losses = 0
    for i in range(grid_ratio):
        for j in range(grid_ratio):
            total_steps = 4001 # Since the whole image is our dataset, this just means 500 gradient descent steps.
            steps_til_summary = 1000
            optim = torch.optim.Adam(lr=1e-4, params=sirens[grid_ratio*i+j].parameters())
            model_input, ground_truth = next(iter(datas[grid_ratio*i+j]))
            model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

            for step in range(total_steps):
                model_output, coords = sirens[grid_ratio*i+j](model_input)
                loss = ((model_output - ground_truth)**2).mean()
                optim.zero_grad()
                loss.backward()
                optim.step()

            losses += loss
            # model_output = model_output.reshape(pic_step, pic_step, 3)
            # output_image[i*pic_step:(i+1)*pic_step, j*pic_step:(j+1)*pic_step, :] = model_output.cpu().detach().numpy()
    return nparams_model, losses/(grid_ratio**2)
