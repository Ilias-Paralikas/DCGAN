import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.utils as vutils
import monai
import torch
from torch.utils.data import Dataset
from discriminator_model import Discriminator
from generator_model import Generator
from weight_initialisation import initialize_weights
from data_loader import MyDataLoader
import sys

outputs_folder = 'outputs'
gen_PATH =os.path.join(outputs_folder,'gen.pt')
disc_PATH = os.path.join(outputs_folder,'disc.pt')
trail_counter_PATH = os.path.join(outputs_folder,'trial_counter.txt')
NUM_EPOCHS  =int(sys.argv[1])
initialize = eval(sys.argv[2])
assert isinstance(initialize, bool)

if not os.path.isfile(trail_counter_PATH):
  f = open(trail_counter_PATH, "w")
  f.write(str(0))
  f.close()
 
BATCH_SIZE = 4
dataset = MyDataLoader()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
NOISE_DIM =100
D_LEARNING_RATE = int(sys.argv[3])  *1e-4  # could also use two lrs, one for gen and one for disc
G_LEARNING_RATE = int(sys.argv[4])  *1e-4# could also use two lrs, one for gen and one for disc


gen = Generator().to(device)
disc =Discriminator().to(device)

if initialize:
    initialize_weights(gen)
    initialize_weights(disc)
    print('initialised weights')
    f = open(trail_counter_PATH, "w")
    f.write(str(0))
    f.close()
else:
    if not os.path.isfile(gen_PATH):
        raise Exception("problema sto gen path")
    else:
        gen.load_state_dict(torch.load(gen_PATH))
        print("read Generator")
    if not os.path.isfile(disc_PATH):
        raise Exception("problema sto disc path")
    else:
        disc.load_state_dict(torch.load(disc_PATH))
        print("read Discriminator")



opt_gen = optim.Adam(gen.parameters(), lr=G_LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=D_LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(1, 100,1, 1, 1).to(device)

if not os.path.isfile(trail_counter_PATH):
    f = open(trail_counter_PATH, "w")
    f.write(str(0))
    f.close()
    trial  = 0
else:
    f = open(trail_counter_PATH, "r")
    trial = int(f.read())
    f.close()

trial = trial+NUM_EPOCHS
for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1,1, 1).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        print("Epoch\t",epoch,'/',NUM_EPOCHS,"\tBatch\t",batch_idx,'/',len(dataloader),"\tLoss D:\t", loss_disc.item(),"\tloss G:\t",loss_gen.item())
    if epoch %10:
        torch.save(gen.state_dict(), gen_PATH)
        torch.save(disc.state_dict(), disc_PATH)
        trial = trial+1

        f = open(trail_counter_PATH, "w")
        f.write(str(trial))
        f.close()
    
    
