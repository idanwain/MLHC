from typing import Dict, List
from db_interface_eicu import DbEicu
from feature import Feature
import pandas as pd
from patient_mimic import PatientMimic
import itertools
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from xgboost import XGBClassifier
import time
import copy
import logging
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from db_interface_mimic import DbMimic
import utils


def main(threshold_vals, kNN_vals, XGB_vals, removal_vals):
    ## Init ###
    db = DbMimic('/Users/user/Documents/University/Workshop/boolean_features_mimic_model_a.csv',
                 '/Users/user/Documents/University/Workshop/extra_features_model_a.csv')
    folds = db.get_folds()
    patient_list_base = db.create_patient_list()

    for product in itertools.product(kNN_vals, XGB_vals, threshold_vals, removal_vals):
        start_time = time.time()
        data = []
        targets = []
        folds_indices = []
        auroc_vals = []
        aupr_vals = []
        patient_list = copy.deepcopy(patient_list_base)

        ### Hyperparameters ###
        threshold = product[2]  # Minimum appearance for feature to be included
        n_neighbors = product[0]  # Neighbors amount for kNN
        xgb_k = product[1]  # Amount of features to return by XGB
        removal_factor = product[3]  # How many negative samples we remove from the training set.
        config = {
            "Threshold": threshold,
            "kNN": n_neighbors,
            "k_XGB": xgb_k,
            "Removal factor": removal_factor
        }
        utils.log_dict(vals=config, msg="Configuration:")

        patient_list = utils.remove_features_by_threshold(threshold, patient_list, db)
        for patient in patient_list:
            targets.append(patient.target)
            for fold in folds:
                if str(patient.hadm_id) in (folds[fold]):
                    folds_indices.append(fold)

        labels_vector = patient_list[0].create_labels_vector()
        for patient in patient_list:
            vector = patient.create_vector_for_patient()
            data.append(vector)

        ### Data imputation ###
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
        data = (imputer.fit_transform(data))

        for test_fold in folds:
            ### Data split ###
            X_train, y_train, X_test, y_test = utils.split_data_by_folds(data, targets, folds_indices, test_fold,
                                                                         removal_factor)
            print(len(X_train))

            ### Feature selection ###
            model = XGBClassifier()
            model.fit(np.asarray(X_train), y_train)
            top_K_xgb = utils.get_top_K_features_xgb(labels_vector, model.feature_importances_.tolist(), k=xgb_k)
            X_train = utils.create_vector_of_important_features(X_train, top_K_xgb)
            X_test = utils.create_vector_of_important_features(X_test, top_K_xgb)

            ### Model fitting ###
            clf_forest = RandomForestClassifier()
            clf_forest.fit(X_train, y_train)

            ### Performance assement ##
            roc_val, pr_val = utils.calc_metrics(y_test, clf_forest.predict(X_test))
            auroc_vals.append(roc_val)
            aupr_vals.append(pr_val)

        ### Log results ###
        utils.log_dict(vals={"AUROC_AVG": np.average(auroc_vals), "AUPR_AVG": np.average(aupr_vals),
                             "AUROC_STD": np.std(auroc_vals), "AUPR_STD": np.std(aupr_vals)}, msg="Run results:")
        utils.log_dict(msg="Running time: " + str(time.time() - start_time))


if __name__ == "__main__":
    threshold_vals = []
    kNN_vals = []
    XGB_vals = []
    removal_vals = []
    for i in range(1, 11):
        kNN_vals.append(i)
        XGB_vals.append(40 + (i * 2))
    for i in range(1, 3):
        removal_vals.append(10 / (10 + i))
    for i in range(1, 6):
        threshold_vals.append(0.1*i)
    main(threshold_vals, kNN_vals, XGB_vals, removal_vals)
