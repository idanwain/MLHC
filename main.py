from typing import Dict, List
from feature import Feature
import pandas as pd
from patient import Patient
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
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
        transfers_before_target, ethnicity, insurance, diagnosis, symptoms = db.get_extra_features_by_hadm_id(hadm_id)
        estimated_age, gender, target = db.get_metadata_by_hadm_id(hadm_id)
        event_list = db.get_events_by_hadm_id(hadm_id)
        boolean_features = db.get_boolean_features_by_hadm_id(hadm_id)
        patient = Patient(hadm_id, estimated_age, gender, ethnicity, transfers_before_target, insurance, diagnosis, symptoms, target, event_list, boolean_features)
        patient_list.append(patient)
        i += 1
    return patient_list


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


def get_top_K_features_xgb(labels_vector, feature_importance: list, k=50):
    indices = []
    list_cpy = feature_importance.copy()
    for i in range(k):
        index = np.argmax(list_cpy)
        indices.append(index)
        list_cpy.pop(index)

    #Print list of features, can be removed
    print("Top %s features according to XGB:" % k)
    for i in indices:
        print("Feature: %s, Importance: %s" % (labels_vector[i], feature_importance[i]))
    return indices


def create_vector_of_important_features(data, features: list):
    """
    Given the top K important features, remove unimportant features.
    :param features:
    :param data: Vectors of features
    :param featurs: top K important features, given by their index
    :return: Set of vectors conatining only relevant features
    """
    new_training_data = []
    for vector in data:
        new_vector = []
        for index in features:
            new_vector.append(vector[index])
        new_training_data.append(new_vector)
    new_training_data = np.asarray(new_training_data)
    return new_training_data


def split_data(data, labels):
    """
    Splits the data into Traning data and test data. Same for the labels.
    The split is currently done hard coded and set to 70% of the data.

    :param data: Array of vectors
    :param labels: Binary vector
    :return: X_train,y_train,X_test,y_test - traning and test data and labels.
    """

    # Doing this so i can randomize the data without losing relation to labels
    joined_data = []
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    pos = []
    neg = []
    data_len = len(data)
    for i in range(data_len):
        joined_data.append((data[i], labels[i]))
    np.random.shuffle(joined_data)
    for vector in joined_data:
        if(vector[1] == 0):
            neg.append(vector)
        else:
            pos.append(vector)
    pos_split_index = int(len(pos)*0.7)
    neg_split_index = int(len(neg) * 0.7)
    train = neg[:neg_split_index] + pos[:pos_split_index]
    test = neg[neg_split_index:] + pos[pos_split_index:]
    # train = joined_data[:int(len(joined_data)*0.7)]
    # test = joined_data[int(len(joined_data)*0.7):]
    for vector in train:
        X_train.append(list(vector[0]))
        y_train.append(vector[1])
    for vector in test:
        X_test.append(list(vector[0]))
        y_test.append(vector[1])
    return X_train,y_train,X_test,y_test


def main():
    data = []
    targets = []
    # db = DB("C:/tools/boolean_features.csv", "C:/tools/extra_features.csv", "C:/tools/feature_mimic_cohort.csv")
    db = DB()
    folds = db.get_folds()
    patient_list = create_patient_list(db)
    patient_list = remove_features_by_threshold(threshold, patient_list, db)
    print(patient_list[0].events)
    for patient in patient_list:
        targets.append(patient.target)
    labels_vector = patient_list[0].create_labels_vector()
    for patient in patient_list:
        vector = patient.create_vector_from_event_list()
        data.append(vector)
    imputer = KNNImputer(n_neighbors=10, weights="uniform")
    data = (imputer.fit_transform(data))
    model = XGBClassifier()
    model.fit(data, targets)
    top_K_xgb = get_top_K_features_xgb(labels_vector, model.feature_importances_.tolist(),k=50)
    data = create_vector_of_important_features(data, top_K_xgb)
    X_train,y_train,X_test,y_test = split_data(data, targets)
    print(X_train,y_train)
    clf_forest = RandomForestClassifier()
    clf_forest.fit(X_train,y_train)
    for i in range(len(X_test)):
        val = clf_forest.predict([X_test[i]])
        print("Predicted: %s. Actual: %s"%(val,y_test[i]))


if __name__ == "__main__":
    main()
