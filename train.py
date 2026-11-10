import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
#from torchsummary import summary
import torch.optim as optim
from time import time
from lib.model.EncDecModel import EncDec
# from lib.model.DilatedNetModel import DilatedNet
# from lib.model.UNetModel import UNet, UNet2
from lib.losses import BCELoss, WeightedBCELoss, FocalLoss

# Dataset
from seg_data import get_datasets
batch_size = 6
ds = get_datasets("ph2")
train_loader = DataLoader(ds["train"], batch_size=batch_size,shuffle = True, num_workers = 3)
val_loader = DataLoader(ds["val"],batch_size = batch_size, shuffle = False, num_workers = 3)
test_loader = DataLoader(ds["test"],batch_size = batch_size, shuffle = False, num_workers= 3)

print(f"Loaded {len(ds['train'])} train / {len(ds['val'])} val / {len(ds['test'])} test images")

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EncDec().to(device)
#model = UNet().to(device) # TODO
#model = UNet2().to(device) # TODO
#model = DilatedNet().to(device) # TODO
#summary(model, (3, 256, 256))
learning_rate = 0.001
opt = optim.Adam(model.parameters(), learning_rate)

loss_fn = BCELoss()
#loss_fn = DiceLoss() # TODO
#loss_fn = FocalLoss() # TODO
#loss_fn = BCELoss_TotalVariation() # TODO
epochs = 20

# Training loop
X_test, Y_test, _ = next(iter(test_loader))
model.train()  # train mode
for epoch in range(epochs):
    tic = time()
    print(f'* Epoch {epoch+1}/{epochs}')

    avg_loss = 0
    for X_batch, y_true, _ in train_loader:
        X_batch = X_batch.to(device)
        y_true = y_true.to(device)

        # set parameter gradients to zero
        opt.zero_grad()

        # forward
        y_pred = model(X_batch)
        # IMPORTANT NOTE: Check whether y_pred is normalized or unnormalized
        # and whether it makes sense to apply sigmoid or softmax.
        loss = loss_fn(y_pred, y_true)  # forward-pass
        loss.backward()  # backward-pass
        opt.step()  # update weights

        # calculate metrics to show the user
        avg_loss += loss.item() / len(train_loader)

    # IMPORTANT NOTE: It is a good practice to check performance on a
    # validation set after each epoch.
    #model.eval()  # testing mode
    #Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
    print(f' - loss: {avg_loss}')

# Save the model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/encdec_ph2.pt")
print("Training has finished!")