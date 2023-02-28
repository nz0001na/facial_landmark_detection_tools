#/*
#  This code is for facial bbox detection
# */

import numpy as np
import argparse
import cv2
import dlib
import imutils
import os
import csv

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((62, 2), dtype=dtype)

    for i in range(62):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


ro = '/home/zn/0_self_collected/'
src_path = ro + '3_cleaned_png/'
dst_path = ro + '3_cleaned_png_bbox.csv'

predictor_file = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

landmark_list = []
landmark_list.append(['img_path', 'x', 'y', 'x1', 'y2', 'width', 'height'])

flist = os.listdir(src_path)
for n in range(len(flist)):
    folder = flist[n]
    print(n)
    namelist = os.listdir(src_path + folder + '/')
    for p in range(len(namelist)):
        name = namelist[p]
        img_list = os.listdir(src_path + folder + '/' + name + '/')
        for q in range(len(img_list)):
            img_name = img_list[q]
            image = cv2.imread(src_path + folder + '/' + name + '/' + img_name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            # may multiple faces in an image
            if len(rects) < 1:
                continue
            rect = rects[0]

            (x, y, w, h) = rect_to_bb(rect)
            x1 = x+w
            y1 = y+h
            landmark_list.append([folder + '/' + name + '/' + img_name, str(x),str(y),str(x1),str(y1),str(w),str(h)])


with open(dst_path, 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(landmark_list)
