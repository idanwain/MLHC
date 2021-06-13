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

    def __init__(self, hadm_id, estimated_age, gender, ethnicity, transfers_before_target, insurance, diagnosis,
                 symptoms, target, events_list, boolean_features):
        self.hadm_id = hadm_id
        self.estimated_age = estimated_age
        self.gender = gender
        self.ethnicity = ethnicity
        self.transfers_before_target = transfers_before_target
        self.insurance = insurance
        self.diagnosis = diagnosis
        self.symptoms = self.disassemble_symptoms(symptoms=symptoms)
        if target == "negative" or target == "Appropriate":
            self.target = 0
        else:
            self.target = 1
        self.events = events_list
        self.boolean_features = boolean_features

    def create_vector_for_patient(self, labels=None, objective_c=False):
        if labels is None:
            labels = self.events
        features_vector = []
        for label in labels:
            features_vector += self.get_essence_values_for_label(label)
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
            return [np.nan] * 4 + [0] + [np.nan]*2
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
            np.average(raw_data[-5:])
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
        bits = []
        while symptoms > 0:
            bits.append(int(symptoms % 2))
            symptoms = int(symptoms / 2)
        while len(bits) != 7:
            bits.append(0)
        return bits
