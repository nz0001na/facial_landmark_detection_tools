'''
    This code is to detect 5 facial points
    Envs: 'Landmark_mtcnn_27'
'''

from src import detect_faces, show_bboxes
from PIL import Image
import os
import csv


ro = '/home/na/3_ASD_micro_expression/2_data/CAS(ME)^2/'
src_path = ro + '2_selectedpic_data/selectedpic_micro/'
dst_path = ro + '3_selectedpic_landmark/selectedpic_micro_5/'

ids = os.listdir(src_path)
for id in ids:
    sub_ids = os.listdir(src_path + id)
    for subid in sub_ids:
        dst_fold = dst_path + id + '/' + subid + '/'
        if os.path.exists(dst_fold) is False:
            os.makedirs(dst_fold)

        img_list = os.listdir(src_path + id + '/' + subid)
        for img in img_list:
            print(id + '/' + subid + '/' + img)
            img_file = src_path + id + '/' + subid + '/' + img
            I = Image.open(img_file)
            bounding_boxes, landmarks = detect_faces(I)
            if len(landmarks) > 0:
                points = landmarks[0]
                p_list = []
                p_list.append(['x', 'y'])
                for i in range(5):
                    p_list.append([str(points[i]), str(points[i+5])])
                with open(dst_fold + img.split('.')[0] + '.csv', 'wb') as f:
                    ft = csv.writer(f)
                    ft.writerows(p_list)

                img_copy = show_bboxes(I, bounding_boxes, landmarks)
                img_copy.save(dst_fold + img)
print('d')

