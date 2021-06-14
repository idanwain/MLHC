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
            features_vector += self.create_delta_vector(label)
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
            return [np.nan] * 4 + [0] + [np.nan] * 5
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
            np.average(raw_data[-5:]),
            max_val - min_val,
            float(np.quantile(raw_data,[0.25])),
            float(np.quantile(raw_data, [0.75]))
        ]

    def create_delta_vector(self, label):
        raw_data = []
        max_delta = 0
        for feature in self.events[label]:
            raw_data.append((feature.value, feature.time))
        if (len(raw_data) == 0):
            return [np.nan]
        sorted_data = sorted(raw_data, key=lambda tup: tup[1])
        # Get maximum delta between any 2 neighbour samples
        for i in range(len(sorted_data) - 1):
            curr_delta = np.abs(sorted_data[i][0] - sorted_data[i + 1][0])
            if (curr_delta > max_delta):
                max_delta = curr_delta
        return [max_delta]
