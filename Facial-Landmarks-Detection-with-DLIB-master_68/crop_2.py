# /*
# this code is for copping faces with bbox points and resize to a fixed size
# square face
# */

import os
import csv
import cv2

ro = '/media/zn/Elements/1_research/1_MAD/1_Data/4_StyleGAN_Morphs_data/'
dst_path = ro + '6_cropped_resize/'
if os.path.exists(dst_path) is False:
    os.makedirs(dst_path)

src_mark_path = ro + '5_all_landmarks/'
csv_list = os.listdir(src_mark_path)
for i in range(len(csv_list)):
    csv_file = csv_list[i]
    if csv_file == 'londondb_genuine_neutral_passport-scale_15kb.csv':
        continue
    print(csv_file)

    new_folder = csv_file.split('.')[0]
    if os.path.exists(dst_path + new_folder + '/') is False:
        os.makedirs(dst_path + new_folder + '/')

    src_mark_file = src_mark_path + csv_file
    bbox_list = []
    f = csv.reader(open(src_mark_file, 'r'))
    for row in f:
        if row[0] == 'img_name':
            continue
        bbox_list.append(row)

    for j in range(len(bbox_list)):
        if j%100 ==0: print(j)

        item = bbox_list[j]
        img_path = item[0]
        li = img_path.split('/')
        img_name = li[len(li)-1]
        x = int(item[1])
        y = int(item[2])
        x1 = int(item[3])
        y1 = int(item[4])
        w = int(item[5])
        h = int(item[6])
        if w != h:
            h = w
        size = w
        img = cv2.imread(ro + img_path)
        crop_img = img[y:y+h, x:x+w]
        resized = cv2.resize(crop_img, (270, 270), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_path + new_folder + '/' + img_name, resized)
