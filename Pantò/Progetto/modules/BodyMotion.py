import cv2
import numpy as np
import json
import os
from tqdm import tqdm
from typing import List

from modules.segmentation import Instance_Segmenter, Bodyparts_Segmenter
from modules.person import Person_A as Person
from modules.id_matcher import Matcher
from modules.result_presenter import BodyMotionResultPresenter

import modules.utils as utils
import modules.face_detection  as face_detection
    
   

def __get_info_frame(frame, detected_people: List[Person]):
    """
    Crea un frame informativo con bounding boxes, maschera di segmentazione delle parti
    del corpo e corner delle persone identificate.

    :param: frame: il frame attuale
    :param: detected people: la lista di persone identificate
    """
    points_colors = {
            "head": (255, 188, 0),
            "left_arm": (154, 255, 0),
            "right_arm":  (0, 255, 94),
            "left_leg":  (0, 180, 255),
            "right_leg":  (154, 0, 255),
            "left_feet":  (255, 0, 222),
            "right_feet": (255, 0, 0)
        }
    
    info_frame = frame.copy()

    for person in detected_people:
        
        x1, y1, x2, y2 = person.box
        #disegna boxes
        
        info_frame = utils.draw_box(info_frame, person.box, (255,0,0))
        
        #disegna corner
        
        if person.bodyparts_points is not None:
            for bodypart, points in person.bodyparts_points.items():
                for point in points:
                    x, y = point[0]
                    cv2.circle(info_frame, (int(x1 + x), int(y1 + y)), 5, points_colors[bodypart], -1)

        text = str(person.id)
        if person.id is None:
            text = "?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, 6)


        padding = 5
        rect_width = text_width + padding * 2
        rect_height = text_height + padding * 2

        # Calcola le coordinate del rettangolo per lo sfondo
        rect_x1 = x1
        rect_y1 = y1 - rect_height - padding
        rect_x2 = rect_x1 + rect_width
        rect_y2 = rect_y1 + rect_height

        # Disegna lo sfondo rosso del testo
        cv2.rectangle(info_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
        cv2.putText(info_frame, text, (rect_x1 + padding, rect_y2 - padding), font, font_scale, (255, 255, 255), 4, cv2.LINE_AA)
        
    return info_frame 


def __resize_masks(masks, new_shape):
    x, y, n = masks.shape
    resized_masks = np.zeros((int(new_shape[0]), int(new_shape[1]), n), dtype=bool)
    
    for i in range(n):
        resized_masks[:,:,i] = cv2.resize(masks[:,:,i].astype(np.uint8), 
                                          (new_shape[1], new_shape[0]), 
                                          interpolation=cv2.INTER_NEAREST)
        
    
    return resized_masks.astype(bool)


def __compute_optical_flow(curr_frame, prev_frame, optical_flow_freq, detected_people: List[Person], resultPresenter: BodyMotionResultPresenter):
    """
    Procedura che calcola l'optical flow per ogni persona della lista e aggiorna il BodyMotionResultPresenter
    """
    for person in detected_people:
        x1, y1, x2, y2 = utils.expand_boxes([person.box], 0.2, curr_frame.shape[1], curr_frame.shape[0])[0]
        x1, y1, x2, y2 = person.box
        curr_image = curr_frame[y1:y2,x1:x2]
        prev_image = prev_frame[y1:y2,x1:x2]

        motion_data = person.compute_new_displacement(prev_image, curr_image, optical_flow_freq)        
        resultPresenter.add_person_detection(person.id, motion_data)

    resultPresenter.add_padding()
    resultPresenter.inc_num_frame()


def __face_recognition(curr_frame, detected_people: List[Person], ResultPresenter: BodyMotionResultPresenter, face_encodings_file: str = None, to_assign: dict = None, max_encodings = 1):
    """
    Effettua il riconoscimento facciale e aggiorna BodyMotionResultPresenter
    :param curr_frame: la lista di persone identificate
    :param detected_people: la lista di persone identificate
    :param face_encodings_file: il percorso del file json per il mtching dei volti, se è None allora viene aggiornato il dizionario "to_assign"
    :param max_encodings: numero massimo di encodings salvati per persona, solo se face_encodings_file = None
    """
    for person in detected_people:

        if person.head_box is not None:
            bb_x1, bb_y1, _, _ = person.box
            hb_x1, hb_y1, hb_x2, hb_y2 = person.head_box

            #converte in coordinate globali(rispetto al frame)
            hb_x1 = hb_x1 + bb_x1
            hb_y1 = hb_y1 + bb_y1
            hb_x2 = hb_x2 + bb_x1
            hb_y2 = hb_y2 + bb_y1

            face_image = curr_frame[hb_y1:hb_y2, hb_x1:hb_x2, :]

            if face_encodings_file is not None:
                if "matching" not in ResultPresenter.detected_people[person.id]:
                    code, landmarks_image = face_detection.face_encode(face_image) 
                    if code is None:
                        continue
                    
                    #tramite un face_recognition viene assegnata una label ad ogni persona identificata   
                    label, dist = face_detection.recognise_person(code, face_encodings_file, 0.5)
                    if label is not None:
                        ResultPresenter.detected_people[person.id]["matching"] = (label, dist)


            #la label viene assegnata manualmente per ogni persona identificata
            elif to_assign is not None:                     
                if person.id not in to_assign:
                    code, landmarks_image = face_detection.face_encode(face_image) 
                    if code is None:  
                        continue 

                    to_assign[person.id] = {
                        "images": [landmarks_image],
                        "encodings": [code]
                    }
                else:
                    if len(to_assign[person.id]["encodings"]) < max_encodings:
                        code, landmarks_image = face_detection.face_encode(face_image) 
                        if code is None:  
                            continue 
                        to_assign[person.id]["images"].append(landmarks_image)
                        to_assign[person.id]["encodings"].append(code)



def compute_motion_values(  video_name: str,
                            output_dir: str, 
                            segmentation_freq: float = 0.2, 
                            optical_flow_freq: float = 3, 
                            face_detection_freq: float = 1, 
                            max_width = None, 
                            face_encodings_file: str = None,
                            ):
    """
    Funzione principale che calcola i valori di movimento di ogni parte del corpo per ogni persona presente in un video
    :param video_name: video
    :param output_dir: cartella output
    :param segmentation_freq: frequenza di ricalcolo della segmentazione
    :param max_height: altezza massima del frame da elaborare, valori più alti danno una precisione maggiore ma richiedono più tempo
    :param face_encodings_file: il file json che contiene gli encodings dei volti, se è "None" viene richiesto l'inserimento manuale delle etichette
    """
    
    instance_segmenter = Instance_Segmenter("resnet50", "average")
    bodyparts_segmenter = Bodyparts_Segmenter()

    video_cap = cv2.VideoCapture(video_name)
    number_of_frame = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = round(video_cap.get(cv2.CAP_PROP_FPS))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    segmentation_step = round(1/segmentation_freq * video_fps)
    optical_flow_step = round(1/optical_flow_freq * video_fps)
    face_detection_step = round(1/face_detection_freq * video_fps)

    #crea cartella di output
    output_dir = os.path.join(output_dir, os.path.basename(video_name).split(".")[0] + "_" + utils.get_datetime())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        output_dir = utils.get_folder_name(output_dir)
        os.makedirs(output_dir)


    ResultPresenter = BodyMotionResultPresenter()
    id_matcher = Matcher(4)

    """
    "to_assign" associa l'id di ogni persona con l'immagine del volto e una lista di encodings. 
    Viene usato per assegnare manualmente una etichetta se il file di encodings dei volti non viene specificato come parametro
    """
    to_assign = dict() 
    detected_people: List[Person] = [] 

    ret, prev_frame = video_cap.read()
    if not ret:
        return


    for frame_count in tqdm(range(number_of_frame-1)):
        ret, frame = video_cap.read()
        if not ret:
            break
        
        #calcola solo la bodyparts segmentation
        if frame_count % segmentation_step == 0:

            #ridimensiona il frame 
            if max_width is not None and max_width < video_width: 
                height = int((max_width * frame.shape[0]) / frame.shape[1])
                resized = cv2.resize(frame, (max_width, height))
            else: resized = frame

            #applica instance segmentation
            masks, boxes = instance_segmenter.instance_segmentation(resized)

            #riporta le masks alla dimensione originale. La bodyparts segmentation è più precisa
            if max_width is not None and max_width < video_width:
                scaleX = frame.shape[1] / resized.shape[1]
                scaleY = frame.shape[0] / resized.shape[0]

                for i, box in enumerate(boxes):
                    x1,y1,x2,y2 = box
                    boxes[i] = int(x1*scaleX), int(y1*scaleY), int(x2*scaleX), int(y2*scaleY)

                masks = __resize_masks(masks, (video_height, video_width))

            #calcola la bodyparts segmentation per ogni persona
            detected_people = []           
            for i in range(masks.shape[2]):
                x1, y1, x2, y2 = boxes[i]
                person_mask = masks[y1:y2,x1:x2,i]
                bodyparts = bodyparts_segmenter.bodyparts_segmentation(frame[y1:y2,x1:x2], person_mask.reshape((person_mask.shape[0], person_mask.shape[1], 1)))
                detected_people.append(Person(None, frame[y1:y2,x1:x2], bodyparts, boxes[i]))    

            detected_people = id_matcher.assign_id(detected_people) 

            for person in detected_people:
                if person.id not in ResultPresenter.detected_people:
                    ResultPresenter.init_person(person.id)

            #salva il primo info_frame
            if frame_count == 0:
                info_frame = __get_info_frame(frame, detected_people)
                cv2.imwrite(os.path.join(output_dir, "info-frame.jpg"), info_frame)
            

        #face_recognition
        if frame_count % face_detection_step == 0 and len(detected_people) > 0:
            __face_recognition(frame, detected_people, ResultPresenter, face_encodings_file, to_assign, 10)


        #calcola solo l'optical flow e aggiorna BodyMotionResultPresenter
        if frame_count % optical_flow_step == 0 and len(detected_people) > 0:           
            __compute_optical_flow(frame, prev_frame, optical_flow_freq, detected_people, ResultPresenter)
            prev_frame = frame
            
            info_frame = __get_info_frame(frame, detected_people)
            utils.imshow(info_frame, 1300, name="info frame")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows() 
    #assegna etichette manualmente e salva il file json con gli encodings
    if face_encodings_file is None:
        file_path = os.path.join(output_dir, "face_encodings.json")
        data = {}

        for id, values in to_assign.items():
            images = values["images"]
            encodings = values["encodings"]
            for i, image in enumerate(images):
                utils.imshow(image, 300, name=f"id: {id} foto {i}")

            cv2.waitKey(0)
            cv2.destroyAllWindows() 
            label = input(f"Inserisci etichetta volto per id = {id}: ")
            if label != "":
                ResultPresenter.detected_people[id]["matching"] = (label,0)
                if label not in data:
                    data[label] = []
                for encoding in encodings: 
                    data[label].append( encoding.tolist() )

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)


    ResultPresenter.join_same_label()
    ResultPresenter.save_as_json(os.path.join(output_dir, "results.json"))
    for id, value in ResultPresenter.detected_people.items():
        if "matching" in value and value['matching'] is not None:
            print(f"{id} = {value['matching']}") 
        else:
            print(f"{id} = ?")      
    print("fatto")

