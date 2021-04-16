from typing import Dict, List
from feature import Feature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

path = "/Users/user/Documents/University/Workshop/features_mimic.csv"
class DB:
    def __init__(self, path: str):
        self.data = pd.read_csv(path)

    def get_hadm_id_list(self) -> list:
        print("Fetching all admission id's")
        hadm_id_list = []
        for id in self.data["hadm_id"]:
            if id not in hadm_id_list:
                hadm_id_list.append(id)
        return hadm_id_list

    def get_labels(self) -> list:
        print("Fetching labels")
        distinct_labels = []
        for label in self.data["label"]:
            if (label not in distinct_labels):
                distinct_labels.append(label)
        return distinct_labels

    def get_events_by_hadm_id(self,hadm_id: str or int) -> Dict[str, List[Feature]]:
        patient_dict = {}
        relevant_rows = self.data.loc[lambda df: df['hadm_id'] == hadm_id, :]
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


