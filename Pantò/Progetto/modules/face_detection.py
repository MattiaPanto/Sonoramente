import json
import face_recognition
import mediapipe as mp
import cv2
import numpy as np

from typing import List


face_detection = mp.solutions.face_detection.FaceDetection()
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)


def __get_centroid(points: List[tuple]):
    """
    calcola il centroide di una lista di punti 
    """
    num_points = len(points)
    if num_points < 1:
        return None
    
    sum_x = 0
    sum_y = 0
    sum_z = 0
    
    # Somma le coordinate x e y di tutti i punti
    for point in points:
        sum_x += point[0]
        sum_y += point[1]
        if len(point) == 3:
            sum_z += point[2]
    
    # Calcola il baricentro
    centroid_x = sum_x / num_points
    centroid_y = sum_y / num_points
    centroid_z = sum_z / num_points
    
    if len(point) == 2:
        return (centroid_x, centroid_y)
    if len(point) == 3:
        return (centroid_x, centroid_y, centroid_z)


def __is_frontal(landmarks, threshold = 0.96):
    """
    verifica che il volto sia frontale
    """
    if landmarks.multi_face_landmarks:
        left_eye = []
        right_eye = []
        for num_point in [253,257,359,463]:
            points = landmarks.multi_face_landmarks[0].landmark[num_point]
            if points.x >= 0 and points.x <= 1 and points.y >= 0 and points.y <= 1:
                left_eye.append( (points.x, points.y , points.z) )
                

        for num_point in [23,27,130,243]:
            points = landmarks.multi_face_landmarks[0].landmark[num_point]
            if points.x >= 0 and points.x <= 1 and points.y >= 0 and points.y <= 1:
                right_eye.append( (points.x, points.y, points.z) )

        
        points = landmarks.multi_face_landmarks[0].landmark[4]
        if points.x >= 0 and points.x <= 1 and points.y >= 0 and points.y <= 1:
            nose = (points.x, points.y, points.z)
        else:
            nose = None

        right_eye = __get_centroid(right_eye)
        left_eye = __get_centroid(left_eye)
        
        if right_eye is None or left_eye is None or nose is None:
            return False

        #controlla che le coordinate x del naso siano comprese tra quelle dei due occhi
        if right_eye[0] >= nose[0] or left_eye[0] <= nose[0]:
            return False
    
  
        if right_eye[0] < nose[0] and left_eye[0] > nose[0]:
            mid_x = (right_eye[0] + left_eye[0])/2
            val = 1-np.abs(mid_x-nose[0])
            if val >= threshold:
                return True
        
        return False



def face_detector(image):
    detection_results = face_detection.process(image)

    if detection_results.detections is not None and len(detection_results.detections) == 1:
        detection = detection_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        points = detection.location_data.relative_keypoints
        # Calcola le coordinate della bounding box rispetto alle dimensioni dell'immagine
        height, width, _ = image.shape
        x1 = int(bbox.xmin * width)
        y1 = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)
        x2 = x1 + w
        y2 = y1 + h

        boxes = (x1,y1,x2,y2)
        return boxes
    return None


def face_encode(img, known_face_location = None):
    """
    Calcola encoding e landmarks in una immagine
    """
    image = img.copy()
    if known_face_location is None:
        boxes = face_detector(image)
    if boxes is not None:
        x1, y1, x2, y2 = boxes
        boxes = (y1,x2,y2,x1)
        landmarks = face_mesh.process(image)
        if __is_frontal(landmarks):      
            code = face_recognition.face_encodings(np.array(image), known_face_locations=[boxes], num_jitters=3, model = "large")[0]
            landmarks = face_recognition.face_landmarks(image, face_locations=[boxes])[0]
            image_with_landmarks = image.copy()
            for landmark_type, landmark_points in landmarks.items():
                for (x, y) in landmark_points:
                    if (x > 0 and x < image_with_landmarks.shape[1]) and (y > 0 and y < image_with_landmarks.shape[0]):
                        cv2.circle(image_with_landmarks, (x, y), 1, (0, 255, 0), -1)
            return code, image_with_landmarks
    return None, None


def recognise_person(unknown, encodings_path, tolerance):
    """
    Cerca il matching con distanza media minima tra l'encoding in input e gli encodings salvati sul file json.

    :param unknown: L'encoding in input
    :param encodings_path: il percorso del file json da confrontare
    :param tolerance: un valore a 0 a 1 che indica la distanza massima da considerare
    """
    if open(encodings_path).read(1):
        with open(encodings_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    
    min_distance = 1
    matched_label = None

    if unknown is None:
        return None, None
    
    for label, encodings in data.items():
        encodings = list(map(np.array, encodings))
        encodings = encodings[:3]
        distances = face_recognition.face_distance(encodings, unknown)

        best_dist = np.mean(distances)
        if best_dist < tolerance and best_dist < min_distance:
            min_distance = best_dist
            matched_label = label

    if matched_label is not None:
        return matched_label, min_distance
    else: 
        return None, None



