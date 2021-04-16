import db_interface
from typing import List
from feature import Feature


class Patient:
    def __init__(self, hadm_id, subj_id, age, gender, ethnicity, target):
        self.hadm_id = hadm_id
        self.subj_id = subj_id
        self.age = age
        self.gender = gender
        self.ethnicity = ethnicity
        self.target = target
        self.events = db_interface.get_events_by_hadm_id(hadm_id)

    def get_feature_from_events(self, label: str or int) -> List[Feature]:
        pass


