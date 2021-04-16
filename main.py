from typing import Dict, List
from feature import Feature
import pandas as pd
from patient import Patient
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from db_interface import DB

relevant_events_path = "/Users/user/Documents/University/Workshop/features_mimic.csv"
threshold = 0.5


def create_patient_list(db):
    """
    Given a DB instance, creates and returns list of all patients
    :param db: DB instance
    :return: list of patients
    """
    hadm_id_list = db.get_hadm_id_list()
    patient_list = []
    for hadm_id in hadm_id_list:
        estimated_age,ethnicity,gender,target = db.get_metadata_by_hadm_id(hadm_id)
        event_list = db.get_events_by_hadm_id(hadm_id)
        patient = Patient(hadm_id,estimated_age,gender,ethnicity,target,event_list)
        patient_list.append(patient)
        return patient_list # Needs to be indented out

def get_features_for_removal(threshold:float,patient_list:list,db):
    """
    Returns a list of features to be removed.
    :param threshold: minimum value of appearances a feature should have
    :param patient_list: list of patients
    :param db: DB instance
    :return: list of features
    """
    data_len = len(patient_list)
    labels = db.get_labels()
    labels_dict = {}
    features_to_be_removed = []
    for label in labels:
        labels_dict[label] = 0
    for patient in patient_list:
        for feature in patient.events:
            if len(patient.events[feature]) > 0:
                labels_dict[feature] += (1/data_len)
    for label in labels:
        if(labels_dict[label] < threshold):
            features_to_be_removed.append(label)
    return features_to_be_removed

def remove_features_by_threshold(threshold:float,patient_list:list,db):
    """
    Removes all features which appear in less than (threshold) of patients
    :param threshold: minimum value of appearances a feature should have
    :param patient_list: list of patients
    :param db: DB instance
    :return: list of patients after removing irrelevant features
    """
    features_to_be_removed = get_features_for_removal(threshold,patient_list,db)
    for patient in patient_list:
        for feature in patient.events.copy():
            if feature in features_to_be_removed:
                del patient.events[feature]
    return patient_list

def main():
    db = DB(relevant_events_path)
    patient_list = create_patient_list(db)
    patient_list = remove_features_by_threshold(threshold,patient_list,db)
    print(patient_list[0].events)













if __name__ == "__main__":
    main()