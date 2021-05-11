from typing import Dict, List
from db_interface_eicu import DbEicu
from feature import Feature
import pandas as pd
from patient_mimic import PatientMimic
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from xgboost import XGBClassifier
import time
import itertools
import copy
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from db_interface_mimic import DbMimic
import utils

user = 'idan'
boolean_features_path = 'C:/tools/boolean_features_mimic_model_b.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/boolean_features_mimic_model_b.csv'
extra_features_path = 'C:/tools/extra_features_model_b.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/extra_features_model_b.csv'
data_path_mimic = 'C:/tools/feature_mimic_cohort_model_b.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/feature_mimic_cohort_model_b.csv'
folds_path = 'C:/tools/folds_mimic_model_b.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/folds_mimic_model_b.csv'

best_run = -1
best_run_val = 0
counter = 1

def main(threshold_vals, kNN_vals, XGB_vals, removal_vals):
    global counter
    db = DbMimic(boolean_features_path,
                 extra_features_path,
                 data_path=data_path_mimic,
                 folds_path=folds_path)

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
            roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr = utils.calc_metrics_roc(clf_forest, X_test, y_test, display_plots=True)
            pr_val, no_skill, lr_recall, lr_precision = utils.calc_metrics_pr(clf_forest, X_test, y_test, display_plots=True)
            auroc_vals.append([roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr])
            aupr_vals.append([pr_val, no_skill, lr_recall, lr_precision])

        utils.plot_graphs(auroc_vals, aupr_vals, counter, 'b')

        counter += 1

        utils.log_dict(vals={"AUROC_AVG": np.average([i[0] for i in auroc_vals]), "AUPR_AVG": np.average([i[0] for i in aupr_vals]),
                             "AUROC_STD": np.std([i[0] for i in auroc_vals]), "AUPR_STD": [i[0] for i in aupr_vals]}, msg="Run results:")
        utils.log_dict(msg="Running time: " + str(time.time() - start_time))
        return np.average([i[0] for i in auroc_vals]) + np.average([i[0] for i in aupr_vals]), counter - 1


if __name__ == "__main__":
    threshold_vals = []
    kNN_vals = []
    XGB_vals = []
    removal_vals = []
    for i in range(1, 11):
        kNN_vals.append(i)
        XGB_vals.append(40 + (i * 2))
    for i in range(1, 3):
        removal_vals.append(5 / (10 + i))
    for i in range(0, 6):
        threshold_vals.append(0.1 * i)
    for a, b, c, d in itertools.product(threshold_vals, kNN_vals, XGB_vals, removal_vals):
        curr_val, run_number = main([a], [b], [c], [d])
        if curr_val > best_run_val:
            best_run_val = curr_val
            best_run = run_number
    utils.log_dict(msg=f"BEST RUN: {best_run}")
