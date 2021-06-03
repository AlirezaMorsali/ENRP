losss = []
import time
t = time.time()
# do stuff
total_steps = 8001 # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 500

optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

for step in range(total_steps):
    model_output, coords = img_siren(model_input)   
    loss = ((model_output - ground_truth)**2).mean()
    dratio = 2
    loss = 0
    dist = int(model_output.shape[1]/(dratio**2))
    # print(model_output.shape, ground_truth.shape, dist)
    # break
    for i in range(dratio):
        for j in range(dratio):
            loss += ((model_output[0, (dratio*i+j)*dist:(dratio*i+j+1)*dist, :] - ground_truth[0, (dratio*i+j)*dist:(dratio*i+j+1)*dist, :])**2).mean()
    
    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f" % (step, loss))
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        axes[0].imshow(model_output.cpu().view(256, 256, 3).detach().numpy()*0.5+0.5)
        plt.show()
        elapsed = time.time() - t
        print(">>> time: ", elapsed)

    losss.append(loss.cpu().detach().numpy())
    optim.zero_grad()
    loss.backward()
    optim.step()

model_big_output = model_output

elapsed = time.time() - t
print(">>> time: ", elapsed)