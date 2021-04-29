from typing import Dict, List
from feature import Feature
import pandas as pd
from patient import Patient
from sklearn.impute import KNNImputer
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from db_interface import DB

threshold = 0.5


def create_patient_list(db):
    """
    Given a DB instance, creates and returns list of all patients
    :param db: DB instance
    :return: list of patients
    """
    hadm_id_list = db.get_hadm_id_list()
    patient_list = []
    i = 0
    for hadm_id in hadm_id_list:
        if i == 500:
            break
        estimated_age, ethnicity, gender, target = db.get_metadata_by_hadm_id(hadm_id)
        event_list = db.get_events_by_hadm_id(hadm_id)
        patient = Patient(hadm_id, estimated_age, gender, ethnicity, target, event_list)
        patient_list.append(patient)
        i += 1
    return patient_list  # Needs to be indented out


def get_features_for_removal(threshold: float, patient_list: list, db):
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
                labels_dict[feature] += (1 / data_len)
    for label in labels:
        if (labels_dict[label] < threshold):
            features_to_be_removed.append(label)
    return features_to_be_removed


def remove_features_by_threshold(threshold: float, patient_list: list, db):
    """
    Removes all features which appear in less than (threshold) of patients
    :param threshold: minimum value of appearances a feature should have
    :param patient_list: list of patients
    :param db: DB instance
    :return: list of patients after removing irrelevant features
    """
    features_to_be_removed = get_features_for_removal(threshold, patient_list, db)
    for patient in patient_list:
        for feature in patient.events.copy():
            if feature in features_to_be_removed:
                del patient.events[feature]
    return patient_list

def get_top_50_features_xgb(labels_vector,feature_importance:list):
    indices = []
    list_cpy = feature_importance.copy()
    for i in range(50):
        index = np.argmax(list_cpy)
        indices.append(index)
        list_cpy.pop(index)

    #Print list of features, can be removed
    print("Top 50 features according to XGB:")
    for i in indices:
        print("Feature: %s, Importance: %s"%(labels_vector[i],feature_importance[i]))
    return indices
def main():
    X_train = []
    y_train = []
    db = DB()
    folds = db.get_folds()
    print(folds)
    patient_list = create_patient_list(db)
    patient_list = remove_features_by_threshold(threshold, patient_list, db)
    print(patient_list[0].events)
    for patient in patient_list:
        y_train.append(patient.target)
    labels_vector = patient_list[0].create_labels_vector()
    for patient in patient_list:
        vector = patient.create_vector_from_event_list()
        X_train.append(vector)
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    X_train = (imputer.fit_transform(X_train))
    model = XGBClassifier()
    model.fit(X_train, y_train)
    top_50_xgb = get_top_50_features_xgb(labels_vector,model.feature_importances_.tolist())



if __name__ == "__main__":
    main()
