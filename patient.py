from typing import List
from feature import Feature
import datetime
import sys
import numpy as np
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
        if target == "negative":
            self.target = 0
        else:
            self.target = 1
        self.events = events_list

    def get_feature_from_events(self, label: str or int) -> List[Feature]:
        pass


    def create_vector_from_event_list(self):
        event_list_vector = []
        for label in self.events:
            event_list_vector += self.get_essence_values_for_label(label)
        return event_list_vector


    def get_essence_values_for_label(self,label):
        """
        Given a label, returns different type of measurements for the label's data.
        NOTE: If the label has no values, its replaced with 4 nans and a 0, with correspondence to the series of features
        :param label: label name
        :return: Array of values, which are ordered as follows:
        Average
        Max value
        Min value
        Latest sample value
        Amount of samples
        """
        avg_val = 0
        max_val = -1
        min_val = sys.maxsize
        latest_sample = {
            "Date": datetime.datetime.now(),
            "Value": 0
        }
        label_vector = []
        number_of_samples = len(self.events[label])
        if(number_of_samples == 0):
            return [np.nan] *4 + [0]
        for feature in self.events[label]:
            avg_val += feature.value
            if feature.value > max_val:
                max_val = feature.value
            if feature.value < min_val:
                min_val = feature.value
            if feature.time > latest_sample["Date"]:
                latest_sample["Date"] = feature.time
                latest_sample["Value"] = feature.value
            label_vector.extend([label + "_avg",label + "_max",label + "_min",label + "_latest",label + "_amount"])
        return [(avg_val/number_of_samples),max_val,min_val,latest_sample["Value"],number_of_samples]

    def create_labels_vector(self):
        ret_vecotr = []
        for label in self.events:
            ret_vecotr.extend([label + "_avg",label + "_max",label + "_min",label + "_latest",label + "_amount",])
        return ret_vecotr
