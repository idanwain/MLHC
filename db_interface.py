from typing import Dict, List
from feature import Feature
import pandas as pd
from patient import Patient
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

relevant_events_path = "/Users/user/Documents/University/Workshop/features_mimic.csv"
folds_path = "/Users/user/Documents/University/Workshop/folds_mimic_model_a.csv"

class DB:
    def __init__(self, data_path="/Users/user/Documents/University/Workshop/features_mimic.csv",folds_path="/Users/user/Documents/University/Workshop/folds_mimic_model_a.csv"):
        self.relevant_events_data = pd.read_csv(data_path)
        self.folds_data = pd.read_csv(folds_path)

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
        print("Fetching labels")
        distinct_labels = []
        for label in self.relevant_events_data["label"]:
            if (label not in distinct_labels):
                distinct_labels.append(label)
        return distinct_labels

    def get_events_by_hadm_id(self,hadm_id: str) -> Dict[str, List[Feature]]:
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
            feature = Feature(time=time,value=value,uom=unit_of_measuere)
            patient_dict[label].append(feature)
        return patient_dict
    def get_metadata_by_hadm_id(self,hadm_id: str):
        """
        Returns a tuple of values which are used as metadata given an hadm_id. Values can be found in Patient object.
        :param hadm_id: id of patient
        :return: tuple of metadata values
        """
        members = [attr for attr in dir(Patient) if not callable(getattr(Patient, attr)) and not attr.startswith("__")]
        values = []
        relevant_rows = self.relevant_events_data.loc[lambda df: df['hadm_id'] == hadm_id, :]
        for member in members:
            values.append(relevant_rows.iloc[0][member])
        return tuple(values)
    def get_folds(self):
        """
        Returns a dictionary of size 5 containing all folds for the cross validation
        :return:
        """
        res = {}
        for row in self.folds_data.iterrows():
            identifier = row[1]["identifier"]
            fold = "fold_"+str(row[1]["fold"])
            print(fold)
            identifier = identifier.split('-')
            hadm_id = identifier[1]
            try:
                res[fold].append(hadm_id)
            except KeyError:
                res.update({fold:[hadm_id]})
        return res



