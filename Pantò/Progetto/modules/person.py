import cv2
import numpy as np

import modules.utils as utils
import modules.bodyparts as bodyparts

from statistics import mode


class Person_A:
    def __init__(self, id: int, image: np.ndarray, bodyparts_mask: np.ndarray, box) -> None:
        self.id = id
        self.image = image
        self.bodyparts_mask = bodyparts_mask
        self.box = box

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_mask = np.zeros(self.bodyparts_mask.shape, dtype=np.uint8)
        gray_mask[self.bodyparts_mask > 0] = 255
        points = np.array(cv2.goodFeaturesToTrack(image_gray, maxCorners=300, qualityLevel=0.007, minDistance=20, mask=gray_mask), dtype=np.int32)

        # è un elenco di tutti i punti rilevati divisi per parte del corpo di appartenenza
        self.bodyparts_points = {
            "head": np.array([point for point in points if bodyparts_mask[point[0][1], point[0][0]] == bodyparts.FACE], dtype=np.float32),
            "left_arm": np.array([point for point in points if bodyparts_mask[point[0][1], point[0][0]] == bodyparts.LEFT_ARM], dtype=np.float32),
            "right_arm":  np.array([point for point in points if bodyparts_mask[point[0][1], point[0][0]] == bodyparts.RIGHT_ARM], dtype=np.float32),
            "left_leg":  np.array([point for point in points if bodyparts_mask[point[0][1], point[0][0]] == bodyparts.LEFT_LEG], dtype=np.float32),
            "right_leg":  np.array([point for point in points if bodyparts_mask[point[0][1], point[0][0]] == bodyparts.RIGHT_LEG], dtype=np.float32),
            "left_feet":  np.array([point for point in points if bodyparts_mask[point[0][1], point[0][0]] == bodyparts.LEFT_FEET], dtype=np.float32),
            "right_feet":  np.array([point for point in points if bodyparts_mask[point[0][1], point[0][0]] == bodyparts.RIGHT_FEET], dtype=np.float32)
        }
           

        #calcola coordinate della head_box
        face_mask = self.bodyparts_mask == 1
        face_mask = utils.clean_mask(face_mask)
        indices = np.argwhere(face_mask)
        if indices.size != 0:
            y_min, x_min = indices.min(axis=0)
            y_max, x_max = indices.max(axis=0) + 1

            self.head_box = utils.expand_boxes([(x_min, y_min, x_max, y_max)], 0.2, image.shape[1], image.shape[0])[0]

            
        else:
            self.head_box = None

    def __points_distances(self, points1, points2):
        """
        calcola le distanze punto a punto tra le due liste di punti.
        :return: la lista delle distanze
        """
        distances = []
        for punto1, punto2 in zip(points1, points2):
            distance = np.sqrt((punto2[0] - punto1[0])**2 + (punto2[1] - punto1[1])**2)
            distances.append(distance)
        return distances


    def __compute_centroid(self, points):
        """
        calcola il centroide di una lista di punti
        """
        import numpy as np

        # Calculate the sum of x-coordinates and y-coordinates
        x_sum = np.sum(points[:, 0])
        y_sum = np.sum(points[:, 1])

        # Calculate the average of x-coordinates and y-coordinates
        x_avg = x_sum / points.shape[0]
        y_avg = y_sum / points.shape[0]

        # Return the centroid coordinates as a tuple
        return int(x_avg), int(y_avg)

    def compute_new_displacement(self, prev_image, curr_image, optical_flow_freq):
        """
        calcola i valori di optical flow tra due frame e aggiorna l'oggetto Person

        :param prev_image: immagine del frame precedente
        :param curr_image: immagine del frame attuale
        :param optical_flow_freq: la frequenza con la quale viene richiamata la funzione
        """ 
        
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        
        bodyparts_data = {
            "head": (0,0,0),
            "left_arm": (0,0,0),
            "right_arm": (0,0,0),
            "left_leg": (0,0,0),
            "right_leg": (0,0,0),
            "left_feet": (0,0,0), 
            "right_feet": (0,0,0)
        }

        lk_params = dict(winSize=(30, 30),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Calcola l'Optical Flow medio per ogni parte del corpo
        for bodypart, points1 in self.bodyparts_points.items():
            if len(points1) == 0:
                continue

            points2, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, self.bodyparts_points[bodypart], None, **lk_params)

            # Seleziona solo i punti di interesse che hanno avuto un buon match
            good_points1 = points1[status == 1]
            good_points2 = points2[status == 1]

            if len(good_points1) == 0:
                continue
            

            #trova il punto che ha avuto lo spostamento maggiore e trova la direzione di spostamento per quel punto
            motion_data = self.__points_distances(good_points1, good_points2) * optical_flow_freq
            max_index = motion_data.index(np.max(motion_data))
            motion_data = motion_data[max_index]

            x_avg, y_avg = self.__compute_centroid(good_points1)

            x_avg += self.box[0]
            y_avg += self.box[1]

            bodyparts_data[bodypart] = (float(round(motion_data, 2)), x_avg, y_avg)
            self.bodyparts_points[bodypart] = np.reshape(good_points2, (good_points2.shape[0], 1, 2))
        
        
        return bodyparts_data
    




class Person_B:
    def __init__(self, id: int, image: np.ndarray, bodyparts_mask: np.ndarray, box) -> None:
        self.id = id
        self.image = image
        self.bodyparts_mask = bodyparts_mask
        self.box = box
        self.bodyparts_points = None

        #calcola coordinate della head_box
        face_mask = self.bodyparts_mask == 1
        face_mask = utils.clean_mask(face_mask)
        indices = np.argwhere(face_mask)
        if indices.size != 0:
            y_min, x_min = indices.min(axis=0)
            y_max, x_max = indices.max(axis=0) + 1

            self.head_box = utils.expand_boxes([(x_min, y_min, x_max, y_max)], 0.2, image.shape[1], image.shape[0])[0]

            
        else:
            self.head_box = None


    def __points_distances(self, points1, points2):
        """
        calcola le distanze punto a punto tra le due liste di punti.
        :return: la lista delle distanze
        """
        distances = []
        for punto1, punto2 in zip(points1, points2):
            distance = np.sqrt((punto2[0] - punto1[0])**2 + (punto2[1] - punto1[1])**2)
            distances.append(distance)
        return distances
    

    def __get_angle(self, point1, point2):
            vect = np.array(point2) - np.array(point1)
            rad = np.arctan2(vect[1], vect[0])
            ang = np.degrees(rad)
            ang = (ang + 360) % 360
            return ang
    
    def __get_xy_direction(self,  point1, point2):
        """
        funzione che calcola la direzione generale del movimento di un insieme di direzioni (su, giù, sx, dx)
        :return direction_x: un valore che può essere 1 se il movimento è verso destra, -1 se è verso sinistra o 0 se non c'è movimento
        :return direction_y: un valore che può essere 1 se il movimento è verso l'alto, -1 se è verso il basso o 0 se non c'è movimento
        """
        direction_x = point2[0] - point1[0]
        direction_y = point2[1] - point1[1]

        if direction_x >= 0:
            direction_x = 1
        else:
            direction_x = -1
        
        if direction_y >= 0:
            direction_y = 1
        else:
            direction_y = -1
        
        return int(direction_x), int(direction_y)
    

    def compute_new_displacement(self, prev_image, curr_image, optical_flow_freq):
        """
        calcola i valori di optical flow tra due frame e aggiorna l'oggetto Person

        :param prev_image: immagine del frame precedente
        :param curr_image: immagine del frame attuale
        :param optical_flow_freq: la frequenza con la quale viene richiamata la funzione
        """ 
        
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        
        bodyparts_data = {
            "head": (0,0,0),
            "left_arm": (0,0,0),
            "right_arm": (0,0,0),
            "left_leg": (0,0,0),
            "right_leg": (0,0,0),
            "left_feet": (0,0,0), 
            "right_feet": (0,0,0)
        }

        gray_mask = np.zeros(self.bodyparts_mask.shape, dtype=np.uint8)
        gray_mask[self.bodyparts_mask > 0] = 255
        points = np.array(cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.007, minDistance=20, mask=gray_mask), dtype=np.int32)

        # è un elenco di tutti i punti rilevati divisi per parte del corpo di appartenenza
        self.bodyparts_points = {
            "head": np.array([point for point in points if self.bodyparts_mask[point[0][1], point[0][0]] == bodyparts.FACE], dtype=np.float32),
            "left_arm": np.array([point for point in points if self.bodyparts_mask[point[0][1], point[0][0]] == bodyparts.LEFT_ARM], dtype=np.float32),
            "right_arm":  np.array([point for point in points if self.bodyparts_mask[point[0][1], point[0][0]] == bodyparts.RIGHT_ARM], dtype=np.float32),
            "left_leg":  np.array([point for point in points if self.bodyparts_mask[point[0][1], point[0][0]] == bodyparts.LEFT_LEG], dtype=np.float32),
            "right_leg":  np.array([point for point in points if self.bodyparts_mask[point[0][1], point[0][0]] == bodyparts.RIGHT_LEG], dtype=np.float32),
            "left_feet":  np.array([point for point in points if self.bodyparts_mask[point[0][1], point[0][0]] == bodyparts.LEFT_FEET], dtype=np.float32),
            "right_feet":  np.array([point for point in points if self.bodyparts_mask[point[0][1], point[0][0]] == bodyparts.RIGHT_FEET], dtype=np.float32)
        }


        lk_params = dict(winSize=(30, 30),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Calcola l'Optical Flow medio per ogni parte del corpo
        for bodypart, points1 in self.bodyparts_points.items():
            if len(points1) == 0:
                continue

            points2, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, self.bodyparts_points[bodypart], None, **lk_params)

            # Seleziona solo i punti di interesse che hanno avuto un buon match
            good_points1 = points1[status == 1]
            good_points2 = points2[status == 1]

            if len(good_points1) == 0:
                continue
            
            
            motion_data = self.__points_distances(good_points1, good_points2) * optical_flow_freq
            max_index = motion_data.index(np.max(motion_data))
            motion_data = motion_data[max_index]
            direction_x, direction_y = self.__get_xy_direction(good_points1[max_index], good_points2[max_index])

            bodyparts_data[bodypart] = (round(np.mean(motion_data), 2), direction_x, direction_y)
        
        
        return bodyparts_data
    