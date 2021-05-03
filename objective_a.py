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
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from db_interface_mimic import DbMimic
import utils

def main():

    ### Hyperparameters ###

    threshold = 0.5     # Minimum appearance for feature to be included
    n_neighbors = 10    # Neighbors amount for kNN
    xgb_k = 50    # Amount of features to return by XG

    ### Data preprocessing - cleaning, imputation and vector creation ###
    start_time = time.time()
    data = []
    targets = []
    db = DbMimic('/Users/user/Documents/University/Workshop/boolean_features_mimic.csv',
            '/Users/user/Documents/University/Workshop/extra_features.csv')
    folds = db.get_folds()
    patient_list = db.create_patient_list()
    patient_list = utils.remove_features_by_threshold(threshold, patient_list, db)
    for patient in patient_list:
        targets.append(patient.target)
    labels_vector = patient_list[0].create_labels_vector()
    for patient in patient_list:
        vector = patient.create_vector_for_patient()
        data.append(vector)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    data = (imputer.fit_transform(data))

    ### Feature selection ###
    model = XGBClassifier()
    model.fit(data, targets)
    top_K_xgb = utils.get_top_K_features_xgb(labels_vector, model.feature_importances_.tolist(), k=xgb_k)
    data = utils.create_vector_of_important_features(data, top_K_xgb)

    for ratio in range(1, 11):

        ### Model fitting ###
        X_train, y_train, X_test, y_test = utils.split_data(data, targets, ratio)
        clf_forest = RandomForestClassifier()
        clf_forest.fit(X_train, y_train)
        # for i in range(len(X_test)):
        #     val = clf_forest.predict([X_test[i]])
        #     print("Predicted: %s. Actual: %s" % (val, y_test[i]))
        # print("[Sanity] Predicted: %s. Actual: %s" % (clf_forest.predict([(X_train[-1])]), y_train[-1]))
        # print("Running time: %s" % (time.time() - start_time))

        ### Performance assement ###
        # err = calc_error(clf_forest, X_test, y_test)
        roc_val,pr_val = utils.calc_metrics(y_test, clf_forest.predict(X_test))
        print("AUROC for iteration %s : %s" % (ratio, roc_val))
        print("AUPR for iteration %s : %s\n" % (ratio, pr_val))



if __name__ == "__main__":
    main()
