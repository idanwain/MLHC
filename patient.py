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


