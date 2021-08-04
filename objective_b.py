from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, SelectFpr, SelectFromModel
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL, atpe
import numpy as np
from functools import partial
import copy
from db_interface_mimic import DbMimic
import utils
from imblearn.under_sampling import TomekLinks, ClusterCentroids, RandomUnderSampler, NearMiss, EditedNearestNeighbours
from hpsklearn import HyperoptEstimator, svc, any_classifier, any_preprocessing, random_forest
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import os
import pickle
from module_1_cohort_creation import create_cohort_training_data
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from imblearn.over_sampling import *
from imblearn.combine import SMOTETomek
from numpy import nan


if os.name == 'posix':
    user = 'roye'
else:
    user = 'idan'
counter = 0
model_type = 'b'

if model_type == 'a':
    boolean_features_path = 'C:/tools/boolean_features_mimic_model_a_train_data.csv' if user == 'idan' \
        else '/Users/user/Documents/University/Workshop/boolean_features_mimic_model_a_train_data.csv'
    extra_features_path = 'C:/tools/extra_features_model_a.csv' if user == 'idan' \
        else '/Users/user/Documents/University/Workshop/extra_features_model_a.csv'
    data_path_mimic = 'C:/tools/external_validation_set_a_train_data.csv' if user == 'idan' \
        else '/Users/user/Documents/University/Workshop/external_validation_set_a_train_data.csv'
    folds_path = 'C:/tools/folds.csv' if user == 'idan' \
        else '/Users/user/Documents/University/Workshop/folds_mimic_model_a.csv'
elif model_type == 'b':
    boolean_features_path = 'C:/tools/boolean_features_mimic_model_b_train_data.csv' if user == 'idan' \
        else '/Users/user/Documents/University/Workshop/boolean_features_mimic_model_b_train_data.csv'
    extra_features_path = 'C:/tools/extra_features_model_b.csv' if user == 'idan' \
        else '/Users/user/Documents/University/Workshop/extra_features_model_b.csv'
    data_path_mimic = 'C:/tools/external_validation_set_b_train_data.csv' if user == 'idan' \
        else '/Users/user/Documents/University/Workshop/external_validation_set_b_train_data.csv'
    folds_path = 'C:/tools/folds_mimic_model_b.csv' if user == 'idan' \
        else '/Users/user/Documents/University/Workshop/folds_mimic_model_b.csv'


def get_fold_indices(patient_list, targets, folds):
    folds_indices = []
    for patient in patient_list:
        targets.append(patient.target)
        for fold in folds:
            if str(patient.hadm_id) in (folds[fold]):
                folds_indices.append(fold)

    return folds_indices


def get_data_vectors(patient_list):
    data = []
    for patient in patient_list:
        vector = patient.create_vector_for_patient()
        data.append(vector)

    return data


def impute_data(X_train, X_test, n_neighbors):
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    return X_train, X_test


def feature_selection(X_train, X_test, y_train, k):
    if k > len(X_train[0]):
        utils.log_dict(msg="Using all features")
        k = len(X_train[0])
    selector = SelectFromModel(estimator=XGBClassifier(),max_features=k)
    selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    indices = selector.get_support(indices=True)
    X_test = selector.transform(X_test)

    return X_train, X_test, indices


def train_model(estim, X_train, y_train):
    try:
        estim.fit(X_train, y_train)
        base_clf = eval(str(estim.best_model()['learner']))
        clf = CalibratedClassifierCV(base_estimator=base_clf)
        clf = clf.fit(X_train, y_train)
    except Exception as e:
        clf = None

    return clf, estim

def get_best_model_and_indices(trails):
    loss = 0
    best_model = None
    indices = []
    for entry in trails.results:
        if(entry['loss'] < loss):
            best_model = entry['clf']
            loss = entry['loss']
            indices = entry['indices']
    return best_model, indices

def save_data_to_disk(model,indices,params):
    model = pickle.dumps(model)
    indices = pickle.dumps(indices)
    params = {'threshold_vals':params['threshold_vals'],'kNN_vals':params['kNN_vals']}
    params = pickle.dumps(params)
    with open('model_' + model_type, 'wb') as model_file:
        model_file.write(model)
    with open('indices_' + model_type, 'wb') as indices_file:
        indices_file.write(indices)
    with open('optimal_values_' + model_type,'wb') as optimal_values_file:
        optimal_values_file.write(params)

def main():
    create_cohort_training_data()
    db = DbMimic(boolean_features_path,
                 data_path=data_path_mimic,
                 folds_path=folds_path)

    folds = db.get_folds()
    patient_list_base = db.create_patient_list()
    space = {
        'threshold_vals': hp.uniform('thershold_val', 0, 1),
        'kNN_vals': hp.choice('kNN_vals', range(1, 15)),
        'XGB1_vals': hp.choice('XGB1_vals', range(1, 60)),
        'XGB2_vals': hp.choice('XGB2_vals', range(1, 60)),
        'XGB3_vals': hp.choice('XGB3_vals', range(1, 60)),
        'XGB4_vals': hp.choice('XGB4_vals', range(1, 60)),
        'XGB5_vals': hp.choice('XGB5_vals', range(1, 60)),
        'balance': hp.choice('balance', [TomekLinks(), RandomUnderSampler()])
    }
    objective_func = partial(objective, patient_list_base=patient_list_base, db=db, folds=folds)
    trials = Trials()
    best = fmin(fn=objective_func, space=space, algo=tpe.suggest, max_evals=5, trials=trials, return_argmin=False)
    best_model, indices = get_best_model_and_indices(trials)
    save_data_to_disk(best_model,indices,best)



def objective(params, patient_list_base, db, folds):
    global counter
    data = []
    targets = []
    auroc_vals = []
    aupr_vals = []
    patient_list = copy.deepcopy(patient_list_base)

    # hyper-parameters
    threshold = params['threshold_vals']  # Minimum appearance for feature to be included
    n_neighbors = params['kNN_vals']  # Neighbors amount for kNN

    # amount of features to return by XGB
    xgb1_k = params['XGB1_vals']
    xgb2_k = params['XGB2_vals']
    xgb3_k = params['XGB3_vals']
    xgb4_k = params['XGB4_vals']
    xgb5_k = params['XGB5_vals']
    xgb_k = [xgb1_k, xgb2_k, xgb3_k, xgb4_k, xgb5_k]

    balance = [params['balance']] * len(folds)

    config = {
        "Threshold": threshold,
        "kNN": n_neighbors,
        "K_best": xgb_k,
        "balance": balance
    }
    utils.log_dict(vals=config, msg="Configuration:")

    patient_list, removed_features = utils.remove_features_by_threshold(threshold, patient_list, db)
    folds_indices = get_fold_indices(patient_list, targets, folds)
    data = get_data_vectors(patient_list)
    labels_vector = utils.create_labels_vector(db, removed_features)

    # normalize data
    data = utils.normalize_data(data)

    for fold_num, test_fold in enumerate(folds):
        # data split
        X_train, y_train, X_test, y_test = utils.split_data_by_folds(data, targets, folds_indices, test_fold)

        # Data imputation
        X_train, X_test = impute_data(X_train, X_test, n_neighbors)

        # Feature selection
        X_train, X_test, indices = feature_selection(X_train, X_test, y_train, xgb_k[fold_num])
        config[f'selected_features_{fold_num}'] = [feature for i, feature in enumerate(labels_vector) if i in indices]

        ### Class balancing ###
        config[f'balance_method_{fold_num}'] = str(balance[fold_num])
        X_train, y_train = balance[fold_num].fit_resample(X_train, y_train)

        # model fitting
        estimator = HyperoptEstimator(classifier=any_classifier('my_clf'),
                                      algo=tpe.suggest,
                                      max_evals=10,
                                      trial_timeout=60)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        clf, estimator = train_model(estimator, X_train, y_train)

        if clf is None:
            auroc_vals = []
            aupr_vals = []
            break

        config[f'best_model_{fold_num}'] = str(estimator.best_model()['learner'])

        # performance assessment
        roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr = utils.calc_metrics_roc(clf, X_test, y_test, X_train, y_train)
        pr_val, no_skill, lr_recall, lr_precision = utils.calc_metrics_pr(clf, X_test, y_test, X_train, y_train)
        auroc_vals.append([roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr])
        aupr_vals.append([pr_val, no_skill, lr_recall, lr_precision])

    counter += 1

    if len(auroc_vals) != len(folds):
        return {
            'loss': 0,
            'status': STATUS_FAIL,
        }

    # save result
    utils.plot_graphs(auroc_vals, aupr_vals, counter, model_type)
    utils.save_conf_file(config, counter, model_type)

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
    return {
        'loss': -1.0 * auroc_avg,
        'status': STATUS_OK,
        'metadata': results,
        'clf': clf,
        'indices' : indices
    }


if __name__ == "__main__":
    main()
