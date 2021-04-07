#! /usr/bin/env python

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from model import End2End
from dataset import TrackDataset

batch_size = 32
epochs = 100

def save_checkpoint(model, epoch):
    if not os.path.exists('./save'):
        os.makedirs('./save')
    torch.save(model.state_dict(), './save/' + 'epoch%d.pth'%epoch)

class BGR2YUV:
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

train_dataset = TrackDataset(1, 9, "./datas")
test_dataset = TrackDataset(10, 10, "./datas")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=2)
tot_batch = len(train_dataloader)
model = End2End()

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = MultiStepLR(optimizer, milestones=[33, 66], gamma=0.25)
criterion = nn.MSELoss()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

print("==> Start training ...")

for epoch in range(epochs):

    train_loss = 0.0
    model.train()

    for i, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device) 
        y_pred = model(X)
        loss = criterion(y_pred, y)
        if type(loss) != torch.Tensor:
            print(X, y, y_pred, loss, X.shape, y.shape, y_pred.shape)
            exit()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data.item()

        if (i+1) % 100 == 0:
            print("Training Epoch: {} | Batch {}/{} | Loss: {} | LR: {}".format(epoch, i+1, tot_batch, train_loss / (i+1), get_lr(optimizer)))

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for i, (X, y) in enumerate(test_dataloader):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            valid_loss += loss.data.item()
            if (i+1) % 100 == 0:
                print("Validation Loss: {}".format(valid_loss / (i+1)))
    
    scheduler.step()
    print("===============================")

    # Save model
    if (epoch+1) % 10 == 0:
        print("==> Save checkpoint ...")
        save_checkpoint(model, epoch+1)