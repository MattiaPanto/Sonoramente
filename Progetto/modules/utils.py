import cv2
import os
import time
import random
import numpy as np
from datetime import datetime
from skimage.measure import label
from typing import List

def resize_image(img, max_height):     
    width = int(img.shape[1] * (max_height / img.shape[0]))
    resized = cv2.resize(img, (width, max_height))
    return resized

def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    colore = (r, g, b)
    return colore

def imshow(image, width, name = "Image"):
    height = int(image.shape[0] * (width / image.shape[1]))
    resized_image = cv2.resize(image, (width, height))
    
    cv2.imshow(name, resized_image)
    cv2.moveWindow(name, 100, 100)


def draw_box(image, box, color = (0,0,255)):
    if box is None:
        return image
    img = image.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    return img

def gray_to_normalized_rgb(image):
    gray_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    mapped = cv2.applyColorMap(gray_img, cv2.COLORMAP_HSV)
    mapped[np.where(gray_img == 0)] = [0, 0, 0]
    return mapped

def get_datetime():
    now = datetime.now()
    result = now.strftime("%d%m%Y%H%M")
    return result


def get_folder_name(folder_name):
    folder_count = 1
    new_folder_name = f"{folder_name}_{folder_count}"
    while os.path.exists(new_folder_name):
        folder_count += 1
        new_folder_name = f"{folder_name}_{folder_count}"
    return new_folder_name


def timeit(function, *args):
    if callable(function):
        startTime = time.time()
        out = function(*args)
        endTime = time.time()
        
        exTime = endTime - startTime
        print(f"Tempo esecuzione '{function.__name__}': {exTime:.6f} secondi")
        return out
    else:
        print("L'argomento passato non è una funzione")


def clean_mask(mask):
    """
    crea una maschera mantenendo solo la regione più estesa 
    """
    labels = label(mask)
    if labels.max() == 0:
        return mask

    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1

    new_mask = np.zeros_like(mask)
    new_mask[labels == largest_label] = mask[labels == largest_label]

    return new_mask


def add_mask(image1, mask):
    if image1.shape != mask.shape:
        raise ValueError("Le dimensioni delle immagini devono essere uguali.")

    if mask.shape[:2] != image1.shape[:2]:
        raise ValueError("La dimensione della maschera deve corrispondere alle dimensioni delle immagini.")

    bin_mask = np.any(mask != [0, 0, 0], axis=-1)
    result_image = image1.copy()
    result_image[bin_mask] = mask[bin_mask]

    return result_image

def merge_masks(masks):
    height, width = masks[0].shape
    num_masks = len(masks)

    merged_mask = np.zeros((height, width), dtype=np.uint8)

    for i in range(num_masks):
        mask = masks[i]
        merged_mask = np.where(mask > 0, mask, merged_mask)
    
    return merged_mask

def dilate_mask(mask, kernel_size, id):
    part_mask = np.zeros_like(mask)
    part_mask[mask == id] = id
    if len(part_mask > 0): 
        dilated_mask = cv2.dilate(part_mask, np.ones((kernel_size, kernel_size), np.uint8), borderType=cv2.BORDER_REPLICATE)
    
        result = merge_masks([mask, dilated_mask])
        return result
    else:
        return mask
    
def expand_boxes(boxes: List[tuple], expansion_percent, max_width, max_height):

    def cut_value(value, max_value):
        if value < 0:  
            value = 0
        
        if value > max_value:  
            value = max_value
        
        return value


    if expansion_percent < 0 or expansion_percent > 1:
        raise ValueError("Il parametro 'expansion_percen' deve essere compreso tra 0 e 1")
    
    expanded_boxes = []
    for box in boxes:
        width = box[2] - box[0]
        height = box[3] - box[1]

        expansion_percent = expansion_percent

        expansion_value = np.mean([width, height]) * expansion_percent

        x1 = int(cut_value(box[0] - expansion_value/2, max_width))
        y1 = int(cut_value(box[1] - expansion_value/2, max_height))
        x2 = int(cut_value(box[2] + expansion_value/2, max_width))
        y2 = int(cut_value(box[3] + expansion_value/2, max_height))

        expanded_boxes.append((x1, y1, x2, y2))
    
    return expanded_boxes

def blur_face(image, box):

    # Applica il blur all'intera immagine
    new = image.copy()
    blurred_image = cv2.GaussianBlur(new, (51, 51), 0)  # Puoi regolare il kernel a seconda dell'effetto di sfocatura desiderato

    x1, y1, x2, y2 = box
    roi = blurred_image[y1:y2, x1:x2]
    new[y1:y2, x1:x2] = roi

    return new