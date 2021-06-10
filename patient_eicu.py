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

    def create_vector_for_patient(self, labels=None, objective_c=False):
        if labels is None:
            labels = self.events
        features_vector = []
        for label in labels:
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
        raw_data = []
        max_val = -1
        min_val = sys.maxsize
        latest_sample = {
            "Date": 0,
            "Value": 0
        }
        number_of_samples = len(self.events[label])
        if number_of_samples == 0:
            return [np.nan] * 4 + [0] + [np.nan] * 2
        for feature in self.events[label]:
            val = float(feature.value)
            raw_data.append(val)
            if val > max_val:
                max_val = val
            if val < min_val:
                min_val = val
            if feature.time > latest_sample["Date"]:
                latest_sample["Date"] = feature.time
                latest_sample["Value"] = val
        return [
            np.average(raw_data),
            max_val,
            min_val,
            latest_sample["Value"],
            number_of_samples,
            np.std(raw_data),
            np.average(raw_data[-5:])
        ]

    def create_labels_vector(self, labels=None, objective_c=False):
        if labels is None:
            labels = self.events
        ret_vector = []
        for label in labels:
            ret_vector.extend([label + "_avg", label + "_max", label + "_min", label + "_latest", label + "_amount"])
        # ret_vector.extend(list(self.boolean_features.keys()))
        return ret_vector

    # def create_vector_for_boolean_features(self):
    #     return list(self.boolean_features.values())
