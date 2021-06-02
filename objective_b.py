from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
import numpy as np
from xgboost import XGBClassifier
import time
import itertools
import copy
from db_interface_mimic import DbMimic
import utils
from imblearn.under_sampling import TomekLinks, ClusterCentroids
from imblearn.over_sampling import *
from imblearn.combine import SMOTETomek

user = 'roye'
boolean_features_path = 'C:/tools/boolean_features_mimic_model_b.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/boolean_features_mimic_model_b.csv'
extra_features_path = 'C:/tools/extra_features_model_b.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/extra_features_model_b.csv'
data_path_mimic = 'C:/tools/feature_mimic_cohort_model_b.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/feature_mimic_cohort_model_b.csv'
folds_path = 'C:/tools/folds_mimic_model_b.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/folds_mimic_model_b.csv'


def main(threshold_vals, kNN_vals, XGB_vals, removal_vals, weights):
    global counter
    counter = 0
    db = DbMimic(boolean_features_path,
                 extra_features_path,
                 data_path=data_path_mimic,
                 folds_path=folds_path)

    folds = db.get_folds()
    patient_list_base = db.create_patient_list()

    for product in itertools.product(kNN_vals, XGB_vals, threshold_vals, removal_vals, weights):
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
        weight = product[4]
        config = {
            "Threshold": threshold,
            "kNN": n_neighbors,
            "k_XGB": xgb_k,
            "Removal factor": removal_factor,
            'weights': weight
        }
        utils.log_dict(vals=config, msg="Configuration:")

        patient_list, removed_features = utils.remove_features_by_threshold(threshold, patient_list, db)
        for patient in patient_list:
            targets.append(patient.target)
            for fold in folds:
                if str(patient.hadm_id) in (folds[fold]):
                    folds_indices.append(fold)

        labels_vector = utils.create_labels_vector(db, removed_features)
        for patient in patient_list:
            vector = patient.create_vector_for_patient()
            data.append(vector)

        ### Data imputation ###
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
        data = (imputer.fit_transform(data))

        for test_fold in folds:
            ### Data split ###
            X_train, y_train, X_test, y_test = utils.split_data_by_folds(data, targets, folds_indices, test_fold)

            ### Feature selection ###
            model = XGBClassifier(use_label_encoder=False)
            model.fit(np.asarray(X_train), y_train)
            top_K_xgb = utils.get_top_K_features_xgb(labels_vector, model.feature_importances_.tolist(), k=xgb_k)
            X_train = utils.create_vector_of_important_features(X_train, top_K_xgb)
            X_test = utils.create_vector_of_important_features(X_test, top_K_xgb)

            ### Class balancing ###
            # over_balancer = BorderlineSMOTE()
            # X_train, y_train = over_balancer.fit_resample(X_train, y_train)
            under_balancer = TomekLinks()
            X_train, y_train = under_balancer.fit_resample(X_train, y_train)

            ### Model fitting ###
            clf1 = DecisionTreeClassifier()
            clf2 = RandomForestClassifier()
            clf3 = KNeighborsClassifier()
            clf = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2), ('knn', clf3)], voting='soft',
                                   weights=weight)
            params = {
                'rf__n_estimators': [50, 175],
                'rf__random_state': [0, 2],
                'dt__random_state': [0, 2],
                'dt__max_depth': [5, 20],
                'knn__n_neighbors': [7, 12],
                'knn__leaf_size': [22, 40]
            }
            grid = GridSearchCV(estimator=clf, param_grid=params)
            grid = grid.fit(X_train, y_train)

            ### Performance assement ##
            roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr = utils.calc_metrics_roc(grid, X_test, y_test)
            pr_val, no_skill, lr_recall, lr_precision = utils.calc_metrics_pr(grid, X_test, y_test)
            auroc_vals.append([roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr])
            aupr_vals.append([pr_val, no_skill, lr_recall, lr_precision])

        utils.plot_graphs(auroc_vals, aupr_vals, counter, 'b')

        counter += 1

        utils.log_dict(vals={"AUROC_AVG": np.average([i[0] for i in auroc_vals]),
                             "AUPR_AVG": np.average([i[0] for i in aupr_vals]),
                             "AUROC_STD": np.std([i[0] for i in auroc_vals]),
                             "AUPR_STD": np.std([i[0] for i in aupr_vals])}, msg="Run results:")
        utils.log_dict(msg="Running time: " + str(time.time() - start_time))
        return np.average([i[0] for i in auroc_vals]) + np.average([i[0] for i in aupr_vals]), counter - 1


if __name__ == "__main__":
    threshold_vals = []
    kNN_vals = []
    XGB_vals = []
    removal_vals = []
    weights = [list(a) for a in list(itertools.product(range(1, 3), range(12, 15), range(1, 3)))]
    for i in range(3, 9):
        XGB_vals.append(40 + (i * 2))
    for i in range(7, 15):
        kNN_vals.append(i)
    for i in range(1, 3):
        removal_vals.append(5 / (10 + i))
    for i in range(0, 10):
        threshold_vals.append(0.1 * i)
    main([0.2], [8], [48], [0.0], [[2, 13, 1]])
