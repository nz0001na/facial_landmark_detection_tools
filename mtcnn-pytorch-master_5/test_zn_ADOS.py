'''
  Envs: 'Landmark_mtcnn_27'
'''

from src import detect_faces, show_bboxes
from PIL import Image
import os
import csv

# '1_Autism'
ro = '/home/na/3_ASD_micro_expression/2_data_ADOS/'
src_path = ro + '5_cleaned_spotted/1_frames/'
dst_path = ro + '6_cleaned_spotted_landmarks/p_5/'

categ_list = os.listdir(src_path)
for catg in categ_list:
    ids = os.listdir(src_path + catg)
    for id in ids:
        sce_ids = os.listdir(src_path + catg + '/' + id)
        for sce in sce_ids:
            frame_list = os.listdir(src_path + catg + '/' + id + '/' + sce)
            for fra_no in frame_list:

                dst_fold = dst_path + catg + '/' + id + '/' + sce + '/' + fra_no + '/'
                if os.path.exists(dst_fold) is True: continue
                if os.path.exists(dst_fold) is False:
                    os.makedirs(dst_fold)

                img_list = os.listdir(src_path + catg + '/' + id + '/' + sce + '/' + fra_no + '/')
                for img in img_list:

                    print(catg + '/' + id + '/' + sce + '/' + fra_no + '/' + img)
                    img_file = src_path + catg + '/' + id + '/' + sce + '/' + fra_no + '/' + img
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

