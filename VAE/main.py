import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from vae import VAE

x = torch.rand(1, 1, 28, 28)

image_size = 28
conv_dims = [32, 64]
fc_dim = 128
latent_dim = 64

batch_size = 1024
epochs = 30

transform=transforms.Compose([
    transforms.ToTensor()
])

dataset1 = datasets.MNIST('./data', train=True, download=False,transform=transform)
dataset2 = datasets.MNIST('./data', train=False,transform=transform)

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

model = VAE(image_size, 1, conv_dims, fc_dim, latent_dim).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print_freq = 200
for epoch in range(epochs):
    print("Start training epoch {}".format(epoch,))
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        recon, mu, log_var = model(images)
        loss = model.compute_loss(images, recon, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
            print("\t [{}/{}]: loss {}".format(i, len(train_loader), loss.item()))


## generate new images by VAE
n_cols, n_rows = 8, 8

sample_zs = torch.randn(n_cols * n_rows, latent_dim)
model.eval()
with torch.no_grad():
    generated_imgs = model.decoder(sample_zs.cuda())
    generated_imgs = generated_imgs.cpu().numpy()
generated_imgs = np.array(generated_imgs * 255, dtype=np.uint8).reshape(n_rows, n_cols, image_size, image_size)
    
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
gs = fig.add_gridspec(n_rows, n_cols)
for n_col in range(n_cols):
    for n_row in range(n_rows):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow(generated_imgs[n_row, n_col], cmap="gray")
        f_ax.axis("off")

plt.savefig('./data/vae_generated_images.png')
plt.show()

## visualize latent features

latent_zs = []
targets = []
for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        with torch.no_grad():
            mu, log_var = model.encoder(images)
        latent_zs.append(mu.cpu().numpy())
        targets.append(labels.numpy())
latent_zs = np.concatenate(latent_zs, 0)
targets = np.concatenate(targets, 0)

n_samples = 1000
sample_idxs = np.random.permutation(len(targets))[:n_samples]

latent_zs = latent_zs[sample_idxs]
targets = targets[sample_idxs]
zs_reduced = TSNE(n_components=2, random_state=2022).fit_transform(latent_zs)

tsne_data = pd.DataFrame({
    "x1": zs_reduced[:,0],
    "x2": zs_reduced[:,1],
    "y": targets})

plt.figure(figsize=(6, 6))
sns.scatterplot(
    x="x1", y="x2",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=tsne_data,
    legend="full",
    alpha=0.3
)   


plt.savefig('./data/visual_latent_feature.png')
plt.show()