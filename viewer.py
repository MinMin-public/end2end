#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import torch
from torch.utils.data import DataLoader
from model import End2End
from dataset import TrackDataset
from collections import deque
import matplotlib.pyplot as plt

# --- drawing functions --- #

def drawgraph(ax, frame_list, y_list, y_pred_list):
    ax.cla()
    ax.plot(frame_list, y_list, color='tab:blue', label="actual")
    ax.plot(frame_list, y_pred_list, color='tab:orange', label="predicted")
    if len(frame_list) < 100:
        ax.axis(xmin=-5, xmax=105, ymin=-20, ymax=20)
    else:
        ax.axis(ymin=-20, ymax=20)
    ax.legend(loc=1)
    ax.set_ylabel("steer angle")
    ax.set_title("Comparison")

def drawsteer(ax, angle, img, title):
    ax.cla()
    (h, w) = img.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    rotated= cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))
    ax.imshow(rotated)
    ax.set_title(title)
    ax.axis('off')

def drawscreen(ax, frame):
    ax.cla()
    ax.axis('off')
    frame = cv2.resize(frame, dsize=(0,0), fx=0.5, fy=0.5)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame)
    ax.set_title("Camera")

def drawtext(ax, i):
    ax.cla()
    ax.text(0.5, 0.5, 'End-to-End Learning for Self-driving Car\n\n Frame {}/{}'.format(i,len(dataloader)),
            horizontalalignment='center', verticalalignment='center', fontsize=12, family='monospace', transform=ax.transAxes)
    ax.axis('off')


# --- load model --- #

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = End2End().to(device)
model.load_state_dict(torch.load("./save/epoch100.pth"))
model.eval()


# --- load dataset --- #

dataset = TrackDataset(4, 4, "./datas")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)


# --- load video --- #

cap = cv2.VideoCapture("./videos/epoch04_front.mkv")

if not cap.isOpened():
    print("cannot find video")
    exit()


# --- load image --- #

steer_img = cv2.imread("steer.png")
steer_img = cv2.resize(steer_img, dsize=(0,0), fx=0.4, fy=0.4)


# --- base figure --- #

plt.figure(figsize=(15,7))
ax1 = plt.subplot2grid((10, 12), (0, 0), colspan=6, rowspan=6)
ax2 = plt.subplot2grid((10, 12), (0, 6), colspan=6, rowspan=6)
ax3 = plt.subplot2grid((10, 12), (7, 6), colspan=3, rowspan=3)
ax4 = plt.subplot2grid((10, 12), (7, 9), colspan=3, rowspan=3)
ax5 = plt.subplot2grid((10, 12), (7, 0), colspan=6, rowspan=3)

plt.tight_layout(w_pad=0, rect=(0.05,0,0.95,0.95))


# --- points to plot --- #

frame_list = deque(maxlen=100)
y_list = deque(maxlen=100)
y_pred_list = deque(maxlen=100)


# --- main loop --- #

with torch.no_grad():
    for i, (X, y) in enumerate(dataloader):
        ret, frame = cap.read()
        if not ret:
            print("video ended")
            break

        X, y = X.to(device), y.to(device).data.item()
        y_pred = model(X).data.item()

        frame_list.append(i)
        y_list.append(y)
        y_pred_list.append(y_pred)

        drawgraph(ax1, frame_list, y_list, y_pred_list)
        drawscreen(ax2, frame)
        drawsteer(ax3, y, steer_img, "actual steer")
        drawsteer(ax4, y_pred, steer_img, "predicted steer")
        drawtext(ax5, i)

        plt.pause(0.003)

    if cap.isOpened():
        cap.release()
    
    cv2.destroyAllWindows()