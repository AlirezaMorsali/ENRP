def train_GSiren_seperated(grid_ratio = 2, min_loss=None):
  total_steps = 2001 # Since the whole image is our dataset, this just means 500 gradient descent steps.
  steps_til_summary = 500

  img_siren = GSiren(in_features=2, out_features=3, hidden_features=256, grid_ratio=grid_ratio, 
                  hidden_layers=3, outermost_linear=True, first_omega_0=10.0, hidden_omega_0=30.0)
  img_siren.cuda()
  count_parameters(img_siren)

  optims = []
  report = {}
  for i in range(grid_ratio):
    for j in range(grid_ratio):
      optims.append(torch.optim.Adam(lr=1e-4, params=img_siren.net[i*grid_ratio+j].parameters()))
      report[str(i)+'_'+str(j)] = []
  model_input, ground_truth = next(iter(dataloader))
  model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
  
  for step in range(total_steps):
      model_output, coords = img_siren(model_input)  
      dratio = grid_ratio
      loss = 0
      length = list(coords.size())[1]
      dist = int(length/grid_ratio)
      for i in range(dratio):
          for j in range(dratio):
            if i!=0 or j!=0:
                model_output, coords = img_siren(model_input)
            loss = ((model_output[i*dist:(i+1)*dist, j*dist:(j+1)*dist, :] - ground_truth[0, i*dist:(i+1)*dist, j*dist:(j+1)*dist, :])**2).mean()
            report[str(i)+'_'+str(j)].append(loss.cpu().detach().numpy())
            cur = i*grid_ratio + j
            optims[cur].zero_grad()
            loss.backward()
            optims[cur].step()

      if not step % steps_til_summary:
          print("Step %d, Total loss %0.6f" % (step, loss))
          fig, axes = plt.subplots(1,3, figsize=(18,6))
          axes[0].imshow(model_output.cpu().view(256, 256, 3).detach().numpy()*0.5+0.5)
          plt.show()
  model_big_output = model_output
  return report