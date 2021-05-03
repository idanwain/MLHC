from typing import List
from feature import Feature
import sys
import numpy as np


class PatientEicu:
    def __init__(self, patient_unit_stay_id, target, events_list):
        self.patient_unit_stay_id = patient_unit_stay_id
        if target == "negative":
            self.target = 0
        else:
            self.target = 1
        self.events = events_list

    def get_feature_from_events(self, label: str or int) -> List[Feature]:
        pass

    def create_vector_for_patient(self):
        features_vector = []
        for label in self.events:
            features_vector += self.get_essence_values_for_label(label)
        # features_vector += self.create_vector_for_boolean_features()
        return features_vector

    def get_essence_values_for_label(self, label):
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
            "Date": 0,
            "Value": 0
        }
        label_vector = []
        number_of_samples = len(self.events[label])
        if number_of_samples == 0:
            return [np.nan] * 4 + [0]
        for feature in self.events[label]:
            avg_val += feature.value
            if feature.value > max_val:
                max_val = feature.value
            if feature.value < min_val:
                min_val = feature.value
            if feature.time > latest_sample["Date"]:
                latest_sample["Date"] = feature.time
                latest_sample["Value"] = feature.value
            label_vector.extend([label + "_avg", label + "_max", label + "_min", label + "_latest", label + "_amount"])
        return [(avg_val/number_of_samples), max_val, min_val, latest_sample["Value"], number_of_samples]

    def create_labels_vector(self):
        ret_vector = []
        for label in self.events:
            ret_vector.extend([label + "_avg", label + "_max", label + "_min", label + "_latest", label + "_amount"])
        # ret_vector.extend(list(self.boolean_features.keys()))
        return ret_vector

    # def create_vector_for_boolean_features(self):
    #     return list(self.boolean_features.values())