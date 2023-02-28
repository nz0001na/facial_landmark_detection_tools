'''
    This code is to crop each frames of spotted micro-expression movements
    by detected bbox
'''

import numpy as np
import argparse
import cv2
import dlib
import imutils
import os
import csv
import random

ro = '/home/na/3_ASD_micro_expression/2_data/CASME2/'
src_path = ro + '2_CASME2_RAW_selected_[spotted]/'
dst_path = ro + '4_CASME2_Cropped_square/'
file_name = ro + 'CASME2_bbox.csv'

f = csv.reader(open(file_name, 'r'))
for row in f:
    if row[0] == 'id': continue
    id = row[0]
    video = row[1]
    x1 = int(row[2])
    y1 = int(row[3])
    x2 = int(row[4])
    y2 = int(row[5])
    dst_fold = dst_path + id + '/' + video + '/'
    if os.path.exists(dst_fold) is False:
        os.makedirs(dst_fold)

    img_list = os.listdir(src_path + id + '/' + video + '/')
    for img in img_list:
        image = cv2.imread(src_path + id + '/' + video + '/' + img)
        crop_img = image[y1:y2, x1:x2]
        resized = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_fold + img[0:len(img)-3] + 'jpg', resized)
