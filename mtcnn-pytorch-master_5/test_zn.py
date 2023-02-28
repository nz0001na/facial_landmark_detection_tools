from src import detect_faces, show_bboxes
from PIL import Image

img = Image.open('frame_001094.jpg')
bounding_boxes, landmarks = detect_faces(img)
img_copy = show_bboxes(img, bounding_boxes, landmarks)

img_copy.save('frame_001094_detect.jpg')
print('d')

# '/home/na/3_ASD_micro_expression/2_data/CAS(ME)^2/2_selectedpic_data/selectedpic_micro/s15_1'