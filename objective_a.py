import itertools
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from xgboost import XGBClassifier
import time
import copy
from db_interface_mimic import DbMimic
import utils

user = 'roye'
boolean_features_path = 'C:/tools/boolean_features_mimic_model_a.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/boolean_features_mimic_model_a.csv'
extra_features_path = 'C:/tools/extra_features_model_a.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/extra_features_model_a.csv'
data_path_mimic = 'C:/tools/feature_mimic_cohort_model_a.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/feature_mimic_cohort_model_a.csv'
folds_path = 'C:/tools/folds_mimic_model_a.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/folds_mimic_model_a.csv'

counter = 1


def main(threshold_vals, kNN_vals, XGB_vals, removal_vals):
    global counter
    ## Init ###
    db = DbMimic(boolean_features_path,
                 extra_features_path,
                 data_path_mimic,
                 folds_path
                 )
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
            roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr = utils.calc_metrics_roc(clf_forest, X_test, y_test)
            pr_val, no_skill, lr_recall, lr_precision = utils.calc_metrics_pr(clf_forest, X_test, y_test)
            auroc_vals.append([roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr])
            aupr_vals.append([pr_val, no_skill, lr_recall, lr_precision])

        utils.plot_graphs(auroc_vals, aupr_vals, counter, 'a')

        counter += 1

        ### Log results ###
        utils.log_dict(vals={"AUROC_AVG": np.average([i[0] for i in auroc_vals]),
                             "AUPR_AVG": np.average([i[0] for i in aupr_vals]),
                             "AUROC_STD": np.std([i[0] for i in auroc_vals]), "AUPR_STD":np.std([i[0] for i in aupr_vals])},
                       msg="Run results:")
        utils.log_dict(msg="Running time: " + str(time.time() - start_time))


if __name__ == "__main__":
    # threshold_vals = []
    # kNN_vals = []
    # XGB_vals = []
    # removal_vals = []
    # for i in range(1, 11):
    #     kNN_vals.append(i)
    #     XGB_vals.append(40 + (i * 2))
    # for i in range(3, 10):
    #     removal_vals.append(10 / (10 + i))
    # for i in range(1, 6):
    #     threshold_vals.append(0.1 * i)
    # main(threshold_vals, kNN_vals, XGB_vals, removal_vals)
    # for a, b, c, d in itertools.product(threshold_vals, kNN_vals, XGB_vals, removal_vals):
    #     main([a], [b], [c], [d])
    main([0.2], [3], [52], [0.5])
