from pixellib.torchbackend.instance import instanceSegmentation
import cv2
from tf_bodypix import api
import numpy as np
from typing import List



import modules.bodyparts as bodyparts
import modules.utils as utils

class Instance_Segmenter:
    def __init__(self, model: str, detection_speed: str = "average") -> None:
        """
        :param model: "resnet50" o "resnet101"
        :param detection_speed: "fast" o "average"
        """ 

        #inizializza il modello pixellib
        self.pixellib_model = instanceSegmentation()

        if detection_speed != "average" and detection_speed != "fast":
            print('Il valore di "detection_speed" può essere "average" o "fast"')
            return
        
        if model == "resnet50":
            self.pixellib_model.load_model("data/models/pointrend_resnet50.pkl", detection_speed=detection_speed)
        elif model == "resnet101":
            self.pixellib_model.load_model("data/models/pointrend_resnet101.pkl", detection_speed=detection_speed, network_backbone="resnet101")
        else:
            print('Il valore di "model" può essere "resnet50" o "resnet101"')
            return
        
    def __get_mean_area(self, boxes):
        sum = 0
        num = 0

        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            sum += area
            num += 1

        if num > 0:
            mean_area = sum / num
            return mean_area
        else:
            return 0
    
    def __delete_small_detections(self, masks, boxes, min_size_percent = 10):
        new_masks = None
        new_boxes = []
        mean_area = self.__get_mean_area(boxes)

        for i,box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if mean_area/100*min_size_percent < area:
                if new_masks is None:
                    # Se l'array delle maschere è vuoto, crea un nuovo array con una singola maschera
                    new_masks = np.expand_dims(masks[:,:, i], axis=2)
                else:
                    new_masks = np.concatenate((new_masks, np.expand_dims(masks[:,:,i], axis=2)), axis=2)

                new_boxes.append(box)

        return new_masks, new_boxes

    def instance_segmentation(self, image: np.ndarray, confidence = 0.8, min_size_percent = 10):
        """
        Applica una instance segmentation su una immagine

        :param image: immagine da elaborare
        :param confidence: soglia minima
        :param min_size_percent: percentuale rispetto alla dimensione media delle boxes, sotto la quale la box viene scartata
        :param return masks: una lista delle maschere binarie calcolate
        :param return boxes: una lista delle coordinate delle boxes trovate
        """ 
        target = self.pixellib_model.select_target_classes(person = True)
        results, output = self.pixellib_model.segmentFrame(image.copy(), segment_target_classes=target)
        
        confidence = confidence*100
        for idx, sc in enumerate(results['scores']):
            if(sc < confidence):
                break
        
        new_mask, new_boxes = self.__delete_small_detections(results['masks'][:, :, :idx], results['boxes'][:idx], min_size_percent)

        return new_mask, new_boxes
    
        


class Bodyparts_Segmenter:
    def __init__(self) -> None:       
        #inizializza modello bodypix
        self.bodypix_model = api.load_model(api.download_model(api.BodyPixModelPaths.RESNET50_FLOAT_STRIDE_16))


    def bodyparts_segmentation(self, image: np.ndarray, mask: np.ndarray):
        """
        Applica una bodyparts segmentation su una immagine

        :param image: immagine da elaborare
        :param mask: maschera binaria
        :param return: una maschera di indici per ogni parte del corpo
        """ 
        COLORS = [
            (bodyparts.FACE,0,0), #left_face
            (bodyparts.FACE,0,0), #right_face
            (bodyparts.LEFT_ARM,0,0), #left_upper_arm_front
            (bodyparts.LEFT_ARM,0,0), #left_upper_arm_back
            (bodyparts.RIGHT_ARM,0,0), #right_upper_arm_front
            (bodyparts.RIGHT_ARM,0,0), #right_upper_arm_back
            (bodyparts.LEFT_ARM,0,0), #left_lower_arm_front
            (bodyparts.LEFT_ARM,0,0), #left_lower_arm_back
            (bodyparts.RIGHT_ARM,0,0), #right_lower_arm_front
            (bodyparts.RIGHT_ARM,0,0), #right_lower_arm_back
            (bodyparts.LEFT_ARM,0,0), #left_hand
            (bodyparts.RIGHT_ARM,0,0), #right_hand
            (bodyparts.TORSO,0,0), #torso_front
            (bodyparts.TORSO,0,0), #torso_back
            (bodyparts.LEFT_LEG,0,0), #left_upper_leg_front
            (bodyparts.LEFT_LEG,0,0), #left_upper_leg_back
            (bodyparts.RIGHT_LEG,0,0), #right_upper_leg_front
            (bodyparts.RIGHT_LEG,0,0), #right_upper_leg_back
            (bodyparts.LEFT_LEG,0,0), #left_lower_leg_front
            (bodyparts.LEFT_LEG,0,0), #left_lower_leg_back
            (bodyparts.RIGHT_LEG,0,0), #right_lower_leg_front
            (bodyparts.RIGHT_LEG,0,0), #right_lower_leg_back
            (bodyparts.LEFT_FEET,0,0), #left_feet
            (bodyparts.RIGHT_FEET,0,0) #right_feet
        ]
        result = self.bodypix_model.predict_single(image)
        colored_mask = cv2.convertScaleAbs(result.get_colored_part_mask(mask, part_colors=COLORS)[:,:,0])
        #colored_mask = cv2.convertScaleAbs(result.get_colored_part_mask(mask)[:,:,0])
        colored_mask = utils.dilate_mask(colored_mask, 30, bodyparts.LEFT_ARM)
        colored_mask = utils.dilate_mask(colored_mask, 30, bodyparts.RIGHT_ARM)
        return colored_mask