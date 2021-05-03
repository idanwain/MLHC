from typing import Dict, List
from feature import Feature
import pandas as pd
from patient_mimic import PatientMimic
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

relevant_events_path = "/Users/user/Documents/University/Workshop/features_mimic.csv"
folds_path = "/Users/user/Documents/University/Workshop/folds_mimic_model_a.csv"


class DbMimic:
    def __init__(self,
                 boolean_features_path,
                 extra_features_path,
                 data_path="/Users/user/Documents/University/Workshop/features_mimic.csv",
                 folds_path="/Users/user/Documents/University/Workshop/folds_mimic_model_a.csv"
                 ):
        self.boolean_features = pd.read_csv(boolean_features_path)
        self.extra_features_data = pd.read_csv(extra_features_path)
        self.relevant_events_data = pd.read_csv(data_path)
        self.folds_data = pd.read_csv(folds_path)
        self.available_labels_in_events = []

    def get_hadm_id_list(self) -> list:
        """
        Returns a list with all distinct hadm_id
        :return: hadm_id list
        """
        print("Fetching all admission id's")
        hadm_id_list = []
        for id in self.relevant_events_data["hadm_id"]:
            if id not in hadm_id_list:
                hadm_id_list.append(id)
        return hadm_id_list

    def get_labels(self) -> list:
        """
        Returns a list with all distinct labels
        :return: labels list
        """
        distinct_labels = []
        if(len(self.available_labels_in_events) == 0):
            for label in self.relevant_events_data["label"]:
                if (label not in distinct_labels):
                    distinct_labels.append(label)
            self.available_labels_in_events = distinct_labels
            return distinct_labels
        else:
            return self.available_labels_in_events

    def get_events_by_hadm_id(self, hadm_id: str) -> Dict[str, List[Feature]]:
        """
        Creates a dictionary of labels and thier values, with a given hadm_id
        :param hadm_id: used to identify the patient
        :return: dictionary of labels and their values (value = Feature object)
        """
        patient_dict = {}
        relevant_rows = self.relevant_events_data.loc[lambda df: df['hadm_id'] == hadm_id, :]
        labels = self.get_labels()
        for label in labels:
            patient_dict[label] = []
        for row in relevant_rows.iterrows():
            label = row[1]["label"]
            time = datetime.strptime(row[1]["charttime"], '%Y-%m-%d %H:%M:%S')
            value = row[1]["valuenum"]
            unit_of_measuere = row[1]["valueuom"]
            feature = Feature(time=time, value=value, uom=unit_of_measuere)
            patient_dict[label].append(feature)
        return patient_dict

    def get_metadata_by_hadm_id(self, hadm_id: str):
        """
        Returns a tuple of values which are used as metadata given an hadm_id. Values can be found in Patient object.
        :param hadm_id: id of patient
        :return: tuple of metadata values
        """
        members = [attr for attr in dir(PatientMimic) if not callable(getattr(PatientMimic, attr)) and not attr.startswith("__")]
        values = []
        relevant_rows = self.relevant_events_data.loc[lambda df: df['hadm_id'] == hadm_id, :]
        for member in members:
            values.append(relevant_rows.iloc[0][member])
        return tuple(values)

    def get_extra_features_by_hadm_id(self, hadm_id: str):
        """
        Returns a tuple of values which are used as extra features given an hadm_id. Values can be found in Patient object.
        :param hadm_id: id of patient
        :return: tuple of metadata values
        """
        relevant_row = self.extra_features_data.loc[lambda df: df['hadm_id'] == hadm_id, :]
        for row in relevant_row.iterrows():
            vals = list(row[1])
            vals.pop(0)
            return tuple(vals)

    def get_folds(self):
        """
        Returns a dictionary of size 5 containing all folds for the cross validation
        :return:
        """
        res = {}
        for row in self.folds_data.iterrows():
            identifier = row[1]["identifier"]
            fold = "fold_" + str(row[1]["fold"])
            identifier = identifier.split('-')
            hadm_id = identifier[1]
            try:
                res[fold].append(hadm_id)
            except KeyError:
                res.update({fold: [hadm_id]})
        return res

    def get_boolean_features_by_hadm_id(self, hadm_id):
        res = {key: 0 for key in self.boolean_features['category']}
        relevant_rows = self.relevant_events_data.loc[lambda df: df['hadm_id'] == hadm_id, :]
        for event in relevant_rows.iterrows():
            if event[1]['itemid'] in list(self.boolean_features['itemid']):
                category = self.boolean_features.loc[lambda df: df['itemid'] == event[1]['itemid'], :]
                res[list(category['category'])[0]] = 1
        return res

    def create_patient_list(self):
        """
        Creates and returns list of all patients
        :return: list of patients
        """
        print("Building patient list from MIMIC...")
        hadm_id_list = self.get_hadm_id_list()
        patient_list = []
        i = 0
        for hadm_id in hadm_id_list:
            if i == 1500:
                break
            transfers_before_target, ethnicity, insurance, diagnosis, symptoms = self.get_extra_features_by_hadm_id(
                hadm_id)
            estimated_age, gender, target = self.get_metadata_by_hadm_id(hadm_id)
            event_list = self.get_events_by_hadm_id(hadm_id)
            boolean_features = self.get_boolean_features_by_hadm_id(hadm_id)
            patient = PatientMimic(hadm_id, estimated_age, gender, ethnicity, transfers_before_target, insurance, diagnosis,
                                   symptoms, target, event_list, boolean_features)
            patient_list.append(patient)
            i += 1
        print("Done")
        return patient_list


