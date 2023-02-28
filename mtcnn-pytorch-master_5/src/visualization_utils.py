from PIL import ImageDraw


def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].

    Returns:
        an instance of PIL.Image.
    """

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)


    # bounding box of each detected face is composed of a array with len=5
    # [x1,y1, x2,y2, probability]
    for b in bounding_boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline='white')

    # landmarks of each detected face is composed of an array with len=10
    # [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]
    # (x1,y1) : left eye center
    # (x2,y2) : right eye center
    # (x3,y3) : nose tip
    # (x4,y4) : left mouth corner
    # (x5,y5) : right mouth corner
    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[i] + 1.0, p[i + 5] + 1.0)
            ], outline='blue')

    return img_copy
