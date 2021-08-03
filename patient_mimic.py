from typing import List
from feature import Feature
import datetime
import sys
import numpy as np
import one_hot_encoding


class PatientMimic:
    estimated_age = None
    gender = None
    target = None

    def __init__(self, hadm_id, estimated_age, gender, symptoms, target, events_list, boolean_features):
        self.hadm_id = hadm_id
        self.estimated_age = estimated_age
        self.gender = gender
        self.symptoms = self.disassemble_symptoms(symptoms=symptoms)
        if target == "negative" or target == "Appropriate":
            self.target = 0
        else:
            self.target = 1
        self.events = {k: events_list[k] for k in sorted(events_list)}
        self.boolean_features = boolean_features

    def create_vector_for_patient(self, labels=None, objective_c=False):
        if labels is None:
            labels = self.events
        features_vector = []
        for label in labels:
            features_vector += self.get_essence_values_for_label(label)
            features_vector += self.create_delta_vector(label)
        if not objective_c:
            features_vector += self.create_vector_for_boolean_features()
            features_vector += self.create_vector_of_categorical_features()
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
        STD
        Average of last 5 samples
        Difference between lowest sample and highest sample
        25th and 75th quartiles
        """
        max_val = -1
        min_val = sys.maxsize
        raw_data = []
        latest_sample = {
            "Date": datetime.datetime.now(),
            "Value": 0
        }
        number_of_samples = len(self.events[label])
        if number_of_samples == 0:
            return [np.nan] * 4 + [0] + [np.nan]*5
        for feature in self.events[label]:
            raw_data.append(feature.value)
            if feature.value > max_val:
                max_val = feature.value
            if feature.value < min_val:
                min_val = feature.value
            if feature.time > latest_sample["Date"]:
                latest_sample["Date"] = feature.time
                latest_sample["Value"] = feature.value
        return [
            np.average(raw_data),
            max_val,
            min_val,
            latest_sample["Value"],
            number_of_samples,
            np.std(raw_data),
            np.average(raw_data[-5:]),
            max_val - min_val,
            float(np.quantile(raw_data, [0.25])),
            float(np.quantile(raw_data, [0.75]))
        ]

    def create_vector_for_boolean_features(self):
        return [self.boolean_features[key] for key in sorted(self.boolean_features)]

    def create_vector_of_categorical_features(self):
        gender_encoding = one_hot_encoding.GENDER_ENCODING[self.gender]
        insurance_encoding = one_hot_encoding.INSURANCE_ENCODING[self.insurance]
        ethnicity_encoding = one_hot_encoding.ETHNICITY_ENCODING[self.ethnicity]
        categorical_vector = gender_encoding + insurance_encoding + ethnicity_encoding + [
            self.transfers_before_target] + self.symptoms
        return categorical_vector

    def disassemble_symptoms(self, symptoms):
        import math
        sparse = [0]*128
        if not math.isnan(symptoms):
            sparse[int(symptoms)] = 1
        return sparse

    def create_delta_vector(self,label):
        raw_data = []
        max_delta = 0
        for feature in self.events[label]:
            raw_data.append((feature.value,feature.time))
        if(len(raw_data) == 0):
            return [np.nan]
        sorted_data = sorted(raw_data, key=lambda tup: tup[1])
        # Get maximum delta between any 2 neighbour samples
        for i in range(len(sorted_data) - 1):
            curr_delta = np.abs(sorted_data[i][0] - sorted_data[i+1][0])
            if(curr_delta > max_delta):
                max_delta = curr_delta
        return [max_delta]

    def get_identifier(self):
        return self.hadm_id

