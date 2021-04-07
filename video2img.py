#! /usr/bin/env python

import glob, os
import cv2

video_files = glob.glob("./videos/*.mkv")

for video_file in video_files:
    video_num = int(filter(str.isdigit, video_file.split('/')[-1]))
    path = './datas/track' + str(video_num) +'/'
    if not os.path.exists(path):
        os.makedirs(path)

    cap = cv2.VideoCapture(video_file)
    if cap.isOpened():
        frame = 0
        while True:
            ret, img = cap.read()
            if not ret:
                print('finished writing %s'%video_file)
                break
            img = cv2.resize(img, (200, 112))[-66:,:]
            cv2.imwrite(path + str(frame) + '.jpg', img)
            frame += 1
        cap.release()
    else:
        print('cannot open the file')
        break