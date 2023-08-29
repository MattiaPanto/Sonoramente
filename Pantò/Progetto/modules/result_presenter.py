import json
from typing import List


class BodyMotionResultPresenter:
    """
    La classe che gestisce il formato di output dei dati, costituita da un dizionario con forma:
    {
        id_persona: {
            "matching": (label, distance)
            "motion_data": {
                "head": [...],
                "left_arm": [...],
                "right_arm": [...],
                "left_leg": [...],
                "right_leg": [...],
                "left_feet": [...], 
                "right_feet": [...]
            }
        }
        ...
        ...
    }

    """
    def __init__(self):
        self.detected_people = dict()
        self.num_frame = 0


    def __join_lists(self, list1, list2, null_val):
        """
        unisce due liste sovrascrivendo le posizioni che contengono null_val
        """

        new_list = []
        for i in range(len(list1)):
            if list1[i] == null_val and list2[i] != null_val:
                new_list.append(list2[i])
            
            elif list1[i] != null_val and list2[i] == null_val:
                new_list.append(list1[i])

            elif list1[i] == null_val and list2[i] == null_val:
                new_list.append(null_val)

            else:
                return None

        return new_list

    def __join_data(self, id_list):
        """
        unisce i dati delle persone con id contenuto nella lista, se dei dati si sovrappongono viene mantenuto il matching con distanza minima
        """

        #trova la persona con il matching migliore
        min_id = id_list[0]
        _,  min_dist = self.detected_people[min_id]["matching"]
        for id in id_list[1:]:
            _, dist = self.detected_people[id]["matching"]
            if dist < min_dist:
                min_dist = dist
                min_id = id
        best_match = self.detected_people[min_id]

        for id in id_list:
            if id == min_id:
                continue

            success = True
            new_bodyparts_motions = dict()
            for bodypart, motion_values in self.detected_people[id]["motion_data"].items():
                new_list = self.__join_lists(best_match["motion_data"][bodypart], motion_values, (-1,0,0))
                if new_list is not None:
                    new_bodyparts_motions[bodypart] = new_list
                else:
                    success = False
                    break

            if success:
                best_match["motion_data"] = new_bodyparts_motions
                del self.detected_people[id]
            else:
                del self.detected_people[id]["matching"]

        
        self.detected_people[min_id] = best_match

    def init_person(self, id):
        self.detected_people[id] = {
            "motion_data": {
                "head": [(-1, 0, 0)] * self.num_frame,
                "left_arm": [(-1, 0, 0)] * self.num_frame,
                "right_arm": [(-1, 0, 0)] * self.num_frame,
                "left_leg": [(-1, 0, 0)] * self.num_frame,
                "right_leg": [(-1, 0, 0)] * self.num_frame,
                "left_feet": [(-1, 0, 0)] * self.num_frame,
                "right_feet": [(-1, 0, 0)] * self.num_frame
            }
        }

    def add_person_detection(self, id, motion_data: dict):
        """
        Aggiunge un valore di movimento per ogni parte del corpo. Viene chiamata ogni volta che deve essere memorizzato
        un valore di optical flow
        :param id: id persona
        :param motion_data: un dizionario contenente un valore di movimento per ogni parte del corpo
        """
        if id in self.detected_people:
            for bodypart, values in motion_data.items():
                self.detected_people[id]["motion_data"][bodypart].append(values)

    def add_padding(self):
        """
        Aggiunge un padding di '-1' per uniformare il numero di valori di movimento di ogni soggetto
        """
        for id, value in self.detected_people.items():
            for bodypart, motion_values in value["motion_data"].items():
                while len(self.detected_people[id]["motion_data"][bodypart]) <= self.num_frame:
                    self.detected_people[id]["motion_data"][bodypart].append((-1,0,0))

    def inc_num_frame(self):
        self.num_frame += 1


    def join_same_label(self):
        """
        Unisce i valori di movimento per i soggetti ai quali è stato trovato un matching con la stessa persona.
        Se i soggetti non possono essere uniti viene eliminato il matching più debole.
        """
        num_same_label = dict()
        for id, value in self.detected_people.items():
            if "matching" not in value:
                continue
            label,_ = value["matching"]
            if label not in num_same_label:
                num_same_label[label] = [id]
            else:
                num_same_label[label].append(id)

        for label, id_list in num_same_label.items():
            if len(id_list) > 1:
                self.__join_data(id_list)


    def save_as_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.detected_people, file, indent=4)    