import pickle
import utils
from db_interface_mimic import DbMimic
import os
from objective_b import get_data_vectors, get_fold_indices, impute_data
import numpy as np

if os.name == 'posix':
    user = 'roye'
else:
    user = 'idan'

counter = 0

"""
This module evaluates the trained model by running 5-folds cross validation.
Saves plot of AUROC and AUPR.
"""


def load_data_from_disk(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def evaluate(model_type):
    boolean_features_path = f'C:/tools/boolean_features_mimic_model_{model_type}_train_data.csv' if user == 'idan' \
        else f'/Users/user/Documents/University/Workshop/boolean_features_mimic_model_{model_type}_train_data.csv'
    data_path_mimic = f'C:/tools/external_validation_set_{model_type}_train_data.csv' if user == 'idan' \
        else f'/Users/user/Documents/University/Workshop/external_validation_set_{model_type}_train_data.csv'
    folds_path = f'C:/tools/model_{model_type}_folds.csv' if user == 'idan' \
        else f'/Users/user/Documents/University/Workshop/folds_mimic_model_{model_type}.csv'
    data_path_eicu = 'C:/tools/model_a_eicu_cohort_training_data.csv' if user == 'idan' \
        else '/Users/user/Documents/University/Workshop/model_a_eicu_cohort_training_data.csv'
    global counter
    targets = []
    auroc_vals = []
    aupr_vals = []
    db = DbMimic(boolean_features_path,
                 mimic_data_path=data_path_mimic,
                 folds_path=folds_path,
                 eicu_data_path=data_path_eicu if model_type == 'a' else None
                 )

    folds = db.get_folds()
    patient_list = db.create_patient_list()

    # hyper-parameters
    optimal_values = load_data_from_disk(f'optimal_values_{model_type}')
    feature_threshold, n_neighbors, patient_threshold = optimal_values['feature_threshold'], optimal_values['kNN_vals'], optimal_values['patient_threshold']
    indices = load_data_from_disk(f'indices_{model_type}')
    balance = load_data_from_disk(f'balance_method_{model_type}')

    patient_list, percentage_removed, total_removed = utils.remove_patients_by_thershold(patient_list,
                                                                                         patient_threshold)

    patient_list, removed_features = utils.remove_features_by_threshold(feature_threshold, patient_list, db)
    folds_indices = get_fold_indices(patient_list, targets, folds)
    data = get_data_vectors(patient_list)
    labels_vector = utils.create_labels_vector(db, removed_features)
    data = utils.create_vector_of_important_features(data, indices)

    for fold_num, test_fold in enumerate(folds):
        # data split
        X_train, y_train, X_test, y_test = utils.split_data_by_folds(data, targets, folds_indices, test_fold)

        # normalize data
        X_train = utils.normalize_data(X_train)
        X_test = utils.normalize_data(X_test)

        # Data imputation
        X_train, X_test = impute_data(X_train, X_test, n_neighbors)

        ### Class balancing ###
        X_train, y_train = balance.fit_resample(X_train, y_train)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # model fitting
        clf = load_data_from_disk(f"pre_trained_model_{model_type}")
        clf.fit(X_train, y_train)

        # performance assessment
        roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr = utils.calc_metrics_roc(clf, X_test, y_test, X_train, y_train)
        pr_val, no_skill, lr_recall, lr_precision = utils.calc_metrics_pr(clf, X_test, y_test, X_train, y_train)
        auroc_vals.append([roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr])
        aupr_vals.append([pr_val, no_skill, lr_recall, lr_precision])

        counter += 1

        # save result
        utils.plot_graphs(auroc_vals, aupr_vals, counter, model_type)

    auroc_avg = np.average([i[0] for i in auroc_vals])
    aupr_avg = np.average([i[0] for i in aupr_vals])
    auroc_std = np.std([i[0] for i in auroc_vals])
    aupr_std = np.std([i[0] for i in aupr_vals])

    results = {
        "AUROC_AVG": auroc_avg,
        "AUPR_AVG": aupr_avg,
        "AUROC_STD": auroc_std,
        "AUPR_STD": aupr_std
    }
    utils.log_dict(vals=results, msg="Run results:")
    print(results)


if __name__ == '__main__':
    evaluate('b')
