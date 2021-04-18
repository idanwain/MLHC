from typing import List
from feature import Feature


class Patient:
    estimated_age = None
    gender = None
    ethnicity = None
    target = None

    def __init__(self, hadm_id, estimated_age, gender, ethnicity, target,events_list):
        self.hadm_id = hadm_id
        self.estimated_age = estimated_age
        self.gender = gender
        self.ethnicity = ethnicity
        self.target = target
        self.events = events_list

    def get_feature_from_events(self, label: str or int) -> List[Feature]:
        pass


    def create_vector_from_event_list(self):
        event_list_vector = []
        for label in self.events:
            avg_val,max_val = self.get_average_and_max_value_for_label(label)
            event_list_vector.append(avg_val)
            event_list_vector.append(max_val)
        return event_list_vector


    def get_average_and_max_value_for_label(self,label):
        avg_val = 0
        max_val = -1
        number_of_samples = len(self.events[label])
        for feature in self.events[label]:
            avg_val += feature.value
            if(feature.value>max_val):
                max_val = feature.value
        return (avg_val/number_of_samples),max_val
