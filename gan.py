# much of the boilerplate code is taken from 
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# however models and dataset are different

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from IPython.display import HTML
#from scipy import io

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "data/music_files"

# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 100

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
#image_size = 9000

# size of training data vector
nx = 900

# Size of z latent vector (i.e. size of generator input)
nz = 50

# Size of hidden layers in generator
ngf = 300

# Size of hidden layers in discriminator
ndf = 50

# Number of training epochs
num_epochs = 500

# Learning rate for optimizers
lr = 0.002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#### Load Data using Dataloader
#### do this after data is saved / converted lmao

#### end dataloader section


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


#play around with the numbs inside plz future me
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear( nz, ngf, bias=True),
            #nn.BatchNorm1d(ngf),
            nn.ReLU(),
            nn.Linear( ngf, ngf, bias=True),
            #nn.BatchNorm1d(ngf),
            nn.ReLU(),
            nn.Linear(ngf, ngf, bias=True),
            #nn.BatchNorm1d(ngf),
            nn.ReLU(),
            nn.Linear(ngf, ngf, bias=True),
            #nn.BatchNorm1d(ngf),
            nn.ReLU(),
            nn.Linear( ngf, nx, bias=True),
            torch.nn.Softmax(dim=0)
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
#print(netG)

# a = np.random.random((1,nz))
# netG.eval()
# print(netG(torch.Tensor(a)))

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Softmax(dim=0),
            nn.Linear( nx, ndf, bias=True),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(.5),
            nn.Linear(ndf, ndf, bias=True),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(.5),
            nn.Linear( ndf, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model

# z = np.random.random((1,nz))
# # netG.eval()
# x = netG(torch.Tensor(z))

# # netD.eval()

# o = netD(x)

# mat_dict = io.loadmat('digits.mat')

# X = mat_dict['X']

X = np.load("music_data.npz")
#print(X['classical'].shape)
#print(np.max(X['classical'][0]))
styles = ['classical','baroque','modern','romantic','addon']
used_styles = ['romantic']
X = torch.Tensor(np.concatenate([X[x][:,:900] for x in used_styles]))
X /= 5
#quit()
dataset = torch.utils.data.TensorDataset(X)

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)



# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr*10)#, momentum = 0.0001)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(.9,.999))#,momentum = 0.001)

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(loader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()

        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        #print(noise.shape)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()

        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        #print(errG.grad)
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        if epoch > 100 and D_G_z2 > .3:
            np.savez("gan_out.npz",fake.detach().cpu().numpy() * 5)
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                #np.savez("gan_out.npz",fake)
            #first_image = np.array(fake[0], dtype='float')
            #pixels = first_image.reshape((28, 28))
            #plt.imshow(pixels, cmap='gray')
            #plt.show()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
