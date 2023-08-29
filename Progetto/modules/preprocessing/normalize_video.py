import cv2
import numpy as np
import subprocess
from modules.segmentation import Instance_Segmenter

def __normalize_angle(angle):
    """
    Normalizza l'angolo in gradi nel range di -180° a 180°.
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def __get_angle(frame_shape: tuple, boxes):
    
    height, width = frame_shape
    sorted_boxes = sorted(boxes, key=lambda box: box[0]) 

    max_distance = -1
    cutting_point_x = None

    for i in range(1, len(sorted_boxes)):
        x1_prev, y1_prev, x2_prev, y2_prev = sorted_boxes[i-1]
        x1_curr, y1_curr, x2_curr, y2_curr = sorted_boxes[i]
        
        distance = x1_curr - x2_prev

        if distance > max_distance:
            max_distance = distance
            cutting_point_x = x2_prev + distance/2

    if cutting_point_x is not None:
        ang = __normalize_angle(360 * cutting_point_x / width)
        return ang
    else:
        return None


def normalize_rotation(input, output_name, padding, fps = 30):
    """  
    Ruota l'angolo di visione del video 360 in modo da evitare di tagliare persone ai bordi e ne riduce la dimensione verticale.

    :param input: path del video360 da analizzare
    :param padding: incremento in px dell'altezza
    """
      
    cap = cv2.VideoCapture(input)
    _, frame = cap.read()
    height, width, _ = frame.shape

    #trova le bboxes nel primo frame
    segmenter = Instance_Segmenter("resnet50", "fast")
    _, boxes = np.array(segmenter.instance_segmentation(frame))
    min_crop = min(box[1] for box in boxes) - padding
    max_crop = max(box[3] for box in boxes) + padding

    if min_crop < 0:
        min_crop = 0
    if max_crop > height:
        max_crop = height

    new_height = max_crop - min_crop
    ang = __get_angle((height, width), boxes)

    ffmpeg_cmd = f"ffmpeg -i {input} -y -preset ultrafast -vf \"crop=in_w:{new_height}:0:{min_crop}, v360=input=e:output=e:yaw={ang}:pitch=0:roll=0:v_flip=0:h_flip=0\" -r {fps} {output_name}"
    
    subprocess.call(ffmpeg_cmd) 





