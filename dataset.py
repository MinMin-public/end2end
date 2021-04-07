#! /usr/bin/env python

import os
import cv2
import torch
import pandas as pd
import numpy as np

import torchvision
from torch.utils.data import Dataset, DataLoader


class BGR2YUV:
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

class TrackDataset(Dataset):

    def __init__(self, start, end, datas_path):

        self.datas_path = datas_path
        self.transform = torchvision.transforms.Compose([BGR2YUV(), torchvision.transforms.ToTensor()])

        dfs = []
        for dataset_num in range(start, end+1):
            csv_file = os.path.join(datas_path, str(dataset_num)+'.csv')
            df = pd.read_csv(csv_file)
            del df['ts_micro']
            df["dataset_num"] = dataset_num
            if 'frame_index' in df:
                df.rename(columns={'frame_index':'frame'}, inplace=True)
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img_path = os.path.join(self.datas_path, 'track' + str(int(data.dataset_num)), str(int(data.frame)) + '.jpg')
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([data.wheel])

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    train_dataset = TrackDataset(1, 9, "./datas")

    dataloader = DataLoader(train_dataset, batch_size=4,
                        shuffle=True, num_workers=2)

    def show_landmarks_batch(sample_batched):
        images_batch, _ = sample_batched
        grid = torchvision.utils.make_grid(images_batch)
        img = grid.numpy().transpose((1, 2, 0))
        img = cv2.cvtColor(np.array(img*255, dtype=np.uint8), cv2.COLOR_YUV2RGB)
        return img

    dataiter = iter(dataloader)

    plt.figure(figsize=(16,3))
    img = show_landmarks_batch(dataiter.next())
    plt.imshow(img)
    plt.title("batch sample")
    plt.axis('off')
    plt.ioff()
    plt.show()