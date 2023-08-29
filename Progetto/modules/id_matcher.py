import cv2
import numpy as np
from typing import List

from modules.person import Person_A as Person


class Matcher:  
    """
    Classe che gestisce l'assegnazione degli id per le persone rilevate tra i frame del video
    """
    def __init__(self, lifetime: int):
        if lifetime < 0:
            raise ValueError("Il valore di 'lifetime' deve essere un intero positivo")
        self.lifetime = int(lifetime)
        self.prev_detection = []
        self.not_matched_people = dict()
        self.num_detections = 0

    def __add_not_matched_person(self, person: Person):
        """
        Se una persona non viene trovata nel frame corrente rimane in memoria per un certo numero di frame analizzati,
        in modo da poter cercare un matching nei frame successivi. Dopo verrà eliminata.
        """
        self.not_matched_people[person.id] = (person, self.lifetime)
    
    def __get_not_matched_people(self):
        not_matched = []
        for _, value in self.not_matched_people.items():
            person, _ = value
            not_matched.append(person)

        return not_matched
    
    def __reduce_not_matched_lifetime(self):
        updated_dict = dict()
        for id, value in self.not_matched_people.items():
            person, lifetime = value
            lifetime -= 1
            if lifetime > 0:
                updated_dict[id] = (person, lifetime)

        self.not_matched_people = updated_dict

    def __find_candidates(self, curr_person: Person, prev_detection: List[Person], min_overlap_percent) -> List[Person]:
        """
        Trova tutte le persone della segmentazione precedente che hanno una sovrapposizione della bbox maggiore di una percentuale
        """
        def same_boxes(box1, box2):
            overlap_area = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
            union_area = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - overlap_area
            overlap_percent = overlap_area / union_area
            return overlap_percent
        
        candidates = []
        for prev_person in prev_detection:
            #se le boundingbox coincidono al 20% la persona diventa una candidata
            overlap_percent = same_boxes(curr_person.box, prev_person.box)
            if overlap_percent > min_overlap_percent:
                candidates.append((prev_person, overlap_percent))

        return candidates
    
    def __find_matching_value(self, curr_image, candidate_image):

        #adatta immagine tagliando orizzontalmente
        curr_height, curr_width, _ = curr_image.shape
        candidate_height, candidate_width, _ = candidate_image.shape

        new_candidate = candidate_image.copy()
        new_curr = curr_image.copy()

        if np.abs(curr_height - candidate_height) < np.abs(curr_width - candidate_width):
            pad = np.abs(int((candidate_height-curr_height)/2))
            if curr_height < candidate_height:
                new_candidate = candidate_image[pad:curr_height+pad,:]
            else:
                new_curr = curr_image[pad:candidate_height+pad,:]
        else:  
            pad = np.abs(int((candidate_width-curr_width)/2))
            if curr_width < candidate_width:
                new_candidate = candidate_image[:, pad:curr_width+pad]
            else:
                new_curr = curr_image[:, pad:candidate_width+pad,:]

        _, max_val, _, _ = cv2.minMaxLoc(cv2.matchTemplate(new_curr, new_candidate, cv2.TM_CCOEFF_NORMED))

        return max_val


    def __match_id(self, curr_detection: List[Person], prev_detection: List[Person]):
        """
        Assegna alla lista di persone del frame attuale il corretto id

        :param curr_detection: La lista di oggetti Person relativa alle persone rilevate nel frame corrente
        :param prev_detection: La lista di oggetti Person relativa alle persone rilevate nel frame precedente
        :return: current_detection con gli id assegnati.
        """

        remaining_to_assign = set(p.id for p in prev_detection)

        for i,curr_person in enumerate(curr_detection):
            """
            Per ogni persona in curr_person vengono valutate le persone in prev_person che hanno una sovrapposizione delle bboxes di almeno 20%.
            Tra queste viene scelto quello ha la media tra valore di template matching e sovrapposizione delle bboxes maggiore.
            """
            candidates = self.__find_candidates(curr_person, prev_detection + self.__get_not_matched_people(), 0.2)

            #assegno l'indice della persona con il valore maggiore di matching tra le candidate
            
            max_idx = -1
            max_similarity = -1

            for candidate, percent in candidates:
                if candidate.id not in remaining_to_assign and candidate.id not in self.not_matched_people:
                    continue

                curr_image = curr_person.image
                candidate_image = candidate.image
                
                max_val = self.__find_matching_value(curr_image, candidate_image)*0.4 + percent*0.6
                
                """print(max_val)
                cv2.imshow("A", curr_image)
                cv2.imshow("B", candidate_image)
                cv2.waitKey(0)"""
                if max_val > max_similarity and max_val > 0.3:
                    max_similarity = max_val
                    max_idx = candidate.id

            if max_idx >= 0:
                #ho matchato una persona del frame precedente e la elimino dalla lista di persona da matchare
                if max_idx not in self.not_matched_people:
                    remaining_to_assign.remove(max_idx) 
                else:
                    del self.not_matched_people[max_idx]

                curr_detection[i].id = max_idx

            else:
                curr_detection[i].id = self.num_detections
                self.num_detections += 1


        
        self.__reduce_not_matched_lifetime()
        for person in prev_detection:
            if person.id in remaining_to_assign:
                self.__add_not_matched_person(person)

        return curr_detection

    def assign_id(self, detected_people: List[Person]):
        """
        Trova il corretto id per le persone in detected_people.
        :param detected_people: la lista di persone a cui assegnare l'id
        """
        
        detected_people = self.__match_id(detected_people, self.prev_detection)
        self.prev_detection = detected_people

        return detected_people