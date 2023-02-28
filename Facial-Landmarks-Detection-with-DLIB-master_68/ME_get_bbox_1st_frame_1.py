'''
    This code is to get the bbox of 1st frames of each spotted micro-expression movements
'''

import numpy as np
import argparse
import cv2
import dlib
import imutils
import os
import csv
import random

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


ro = '/home/na/3_ASD_micro_expression/2_data/CASME2/'
src_path = ro + '2_CASME2_RAW_selected_[spotted]/'

file_name = ro + 'CASME2_bbox.csv'

predictor_file = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

results = []
item = ['id', 'video', 'x1', 'y1', 'x2', 'y2']
results.append(item)

# sub01/EP02_01f/img75.jpg
ids = os.listdir(src_path)
for id in ids:
    videos = os.listdir(src_path + id)
    for video in videos:
        imgs = os.listdir(src_path + id + '/' + video)
        img_list = sorted(imgs)

        image = cv2.imread(src_path + id + '/' + video + '/' + img_list[0])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        # may multiple faces in an image
        if len(rects) < 1:
            continue
        rect = rects[0]
        (x1, y1, w, h) = rect_to_bb(rect)
        x2 = x1+w
        y2 = y1+h
        results.append([str(id), str(video), str(x1), str(y1), str(x2), str(y2)])
        # item = ['id', 'video', 'x1', 'y1', 'x2', 'y2']

with open(file_name, 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(results)
