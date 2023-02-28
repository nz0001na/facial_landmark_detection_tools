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
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


ro = '/home/na/1_MAD/2_Data/2_few_shot_dbs/'
src_path = ro # + '1_Jake_LMA/'
dst_path = '/home/na/1_MAD/2_Data/cropped/'
if os.path.exists(dst_path) is False: os.makedirs(dst_path)

predictor_file = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

flist = [
    # '1_Jake_LMA',
    # '1_Jake_real',
    # '1_Jake_stylegan_sharp'
    # '3_FRGC_stylegan_sharp_512', '3_FRGC_stylegan_sharp_1024',
    '5_self_Facemorpher_1024_1', '5_self_stylegan_1024_1'

         ]
# flist = os.listdir(src_path)
for n in range(len(flist)):
    folder = flist[n]
    if os.path.exists(dst_path+ folder + '/') is False:
        os.makedirs(dst_path+ folder + '/')

    img_list = os.listdir(src_path + folder)
    if os.path.exists(dst_path + folder) is False:
        os.makedirs(dst_path + folder)

    for q in range(len(img_list)):
        img_name = img_list[q]
        print(folder + ':' + img_name)
        image = cv2.imread(src_path + folder + '/' + img_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        # may multiple faces in an image
        if len(rects) < 1:
            continue
        rect = rects[0]

        (x, y, w, h) = rect_to_bb(rect)
        x1 = x+w
        y1 = y+h

        if w != h: h = w
        size = w
        crop_img = image[y:y+h, x:x+w]
        resized = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_path + folder + '/' + img_name + '.png', resized)


