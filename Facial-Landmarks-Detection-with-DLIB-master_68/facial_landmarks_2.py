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

predictor_file = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

ro = '/home/na/1_Face_morphing/2_data/FRGC-Morphs/frgc/'
src_path = ro + 'raw_aligned_512_pairs/'
dst_path = ro + 'raw_aligned_512_pairs_landmarks/'
if os.path.exists(dst_path) is False:
    os.makedirs(dst_path)

flist = os.listdir(src_path)
for folder in flist:
    namelist = os.listdir(src_path + folder + '/')
    if os.path.exists(dst_path + folder) is False:
        os.makedirs(dst_path + folder)

    for name in namelist:
        if os.path.exists(dst_path + folder + '/' + name) is False:
            os.makedirs(dst_path + folder + '/' + name)

        img_list = os.listdir(src_path + folder + '/' + name + '/')
        for img in img_list:
            image = cv2.imread(src_path + folder + '/' + name + '/' + img)
            # image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            landmarks = []
            rects = detector(gray, 1)
            # for (i, rect) in enumerate(rects):
            shape = predictor(gray, rects[0])
            shape = shape_to_np(shape)

            for (x, y) in shape:
                landmarks.append([str(x),str(y)])
                # cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            with open(dst_path + folder + '/' + name + '/' + img[0:len(img)-3] + 'csv', 'w', newline='') as f:
                ft = csv.writer(f)
                ft.writerows(landmarks)
