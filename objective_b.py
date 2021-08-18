from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, SelectFpr, SelectFromModel, chi2
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL, atpe
import numpy as np
from functools import partial
import copy
from db_interface_mimic import DbMimic
import utils
from imblearn.under_sampling import TomekLinks, ClusterCentroids, RandomUnderSampler, NearMiss, EditedNearestNeighbours
from hpsklearn import HyperoptEstimator, sgd, any_classifier, svc, random_forest, ada_boost, extra_trees, knn
from hpsklearn import xgboost_classification as xgb_clf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
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


boolean_features_path = f'C:/tools/boolean_features_mimic_model_{model_type}_train_data.csv' if user == 'idan' \
    else f'/Users/user/Documents/University/Workshop/boolean_features_mimic_model_{model_type}_train_data.csv'
data_path_mimic = f'C:/tools/external_validation_set_{model_type}_train_data.csv' if user == 'idan' \
    else f'/Users/user/Documents/University/Workshop/external_validation_set_{model_type}_train_data.csv'
folds_path = f'C:/tools/model_{model_type}_folds.csv' if user == 'idan' \
    else f'/Users/user/Documents/University/Workshop/folds_mimic_model_{model_type}.csv'
data_path_eicu = 'C:/tools/model_a_eicu_cohort_training_data.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/model_a_eicu_cohort_training_data.csv'

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
    try:
        estimator = HyperoptEstimator(classifier=xgb_clf('xgb_clf'),
                                      algo=tpe.suggest,
                                      max_evals=3,
                                      trial_timeout=120)
        estimator.fit(X_train, y_train)
        clf = eval(str(estimator.best_model()['learner']))
        selector = SelectFromModel(estimator=clf, max_features=k)
    except Exception as e:
        selector = SelectKBest(k=k)
    selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    indices = selector.get_support(indices=True)
    X_test = selector.transform(X_test)

    return X_train, X_test, indices


def train_model(estimator_list, weight, X_train, y_train):
    estimators = [(f'clf{i + 1}', clf) for i, clf in enumerate(estimator_list)]
    voting_clf = VotingClassifier(estimators=estimators, voting='soft', weights=weight)
    pre_trained_clf = copy.deepcopy(voting_clf)
    voting_clf.fit(X_train, y_train)
    return voting_clf, estimator_list, pre_trained_clf


def get_best_model_and_indices(trails):
    loss = 0
    best_model = None
    indices = []
    exclusion = {}
    best_model_pre_trained = None
    balance_method = None
    for entry in trails.results:
        if (entry['loss'] < loss):
            best_model = entry['trained_clf']
            best_model_pre_trained = entry['pre_trained_clf']
            balance_method = entry['balance_method']
            loss = entry['loss']
            indices = entry['indices']
            exclusion = entry['exclusion']
    return best_model, indices, exclusion, best_model_pre_trained, balance_method


def save_data_to_disk(model, indices, params, exclusion, pre_trained_model, balance_method):
    model = pickle.dumps(model)
    indices = pickle.dumps(indices)
    params = {
        'feature_threshold': params['feature_threshold'],
        'kNN_vals': params['kNN_vals'],
        'patient_threshold': params['patient_threshold']
    }
    params = pickle.dumps(params)
    exclusion = pickle.dumps(exclusion)
    pre_trained_model = pickle.dumps(pre_trained_model)
    balance_method = pickle.dumps(balance_method)
    with open('model_' + model_type, 'wb') as model_file:
        model_file.write(model)
    with open('pre_trained_model_' + model_type, 'wb') as pre_trained_model_file:
        pre_trained_model_file.write(pre_trained_model)
    with open('indices_' + model_type, 'wb') as indices_file:
        indices_file.write(indices)
    with open('optimal_values_' + model_type, 'wb') as optimal_values_file:
        optimal_values_file.write(params)
    with open('exclusion_criteria_' + model_type, 'wb') as exclusion_file:
        exclusion_file.write(exclusion)
    with open('balance_method_' + model_type, 'wb') as balance_method_file:
        balance_method_file.write(balance_method)


def estimate_best_model(clf, data_mimic, targets_mimic):
    estimator = HyperoptEstimator(classifier=clf,
                                  algo=atpe.suggest,
                                  max_evals=10,
                                  trial_timeout=120)
    try:
        estimator.fit(data_mimic, targets_mimic)
        res = eval(str(estimator.best_model()['learner']))
    except Exception as e:
        res = RandomForestClassifier()
    return res


def main(given_model_type=None):
    global model_type
    create_cohort_training_data(model_type)
    if given_model_type is not None:
        model_type = given_model_type
    db = DbMimic(boolean_features_path,
                 mimic_data_path=data_path_mimic,
                 folds_path=folds_path,
                 eicu_data_path=data_path_eicu if model_type == 'a' else None
                 )

    folds = db.get_folds()
    patient_list_base = db.create_patient_list()
    space = {
        'feature_threshold': hp.uniform('thershold_val', 0.5, 1),
        'patient_threshold': hp.uniform('patient_threshold', 0.5, 1),
        'kNN_vals': hp.choice('kNN_vals', range(3, 15)),
        'XGB_k': hp.choice('XGB1_vals', range(40, 80)),
        'clf1_weight': hp.choice('clf1_weight', range(1, 30)),
        'clf2_weight': hp.choice('clf2_weight', range(1, 30)),
        'balance': hp.choice('balance', [TomekLinks(), RandomUnderSampler(), BorderlineSMOTE()])
    }
    objective_func = partial(objective, patient_list_base=patient_list_base, db=db, folds=folds)
    trials = Trials()
    best = fmin(fn=objective_func, space=space, algo=tpe.suggest, max_evals=5, trials=trials, return_argmin=False)
    best_model, indices, exclusion, best_model_pre_trained, balance_method = get_best_model_and_indices(trials)
    save_data_to_disk(best_model, indices, best, exclusion, best_model_pre_trained, balance_method)


def objective(params, patient_list_base, db, folds):
    global counter
    targets = []
    auroc_vals = []
    aupr_vals = []
    patient_list = copy.deepcopy(patient_list_base)

    # hyper-parameters
    feature_threshold = params['feature_threshold']  # Minimum appearance for feature to be included
    n_neighbors = params['kNN_vals']  # Neighbors amount for kNN
    patient_threshold = params['patient_threshold']  # Percentage of missing features of patient
    clf1_weight = params['clf1_weight']
    clf2_weight = params['clf2_weight']
    weight = [clf1_weight, clf2_weight]
    # amount of features to return by XGB
    xgb_k = params['XGB_k']
    balance = [params['balance']] * len(folds)

    config = {
        "Feature Threshold": feature_threshold,
        "Patient Thershold": patient_threshold,
        "kNN": n_neighbors,
        "K_best": xgb_k,
        "balance": balance,
        "weight": weight
    }
    utils.log_dict(vals=config, msg="Configuration:")

    patient_list, percentage_removed, total_removed = utils.remove_patients_by_thershold(patient_list,
                                                                                         patient_threshold)

    patient_list, removed_features = utils.remove_features_by_threshold(feature_threshold, patient_list, db)
    folds_indices = get_fold_indices(patient_list, targets, folds)
    data = get_data_vectors(patient_list)
    labels_vector = utils.create_labels_vector(db, removed_features)

    # fixed fold = fixed train data & fixed test data
    fold_num = 0
    test_fold = 'fold_1'
    # data split
    X_train, y_train, X_test, y_test = utils.split_data_by_folds(data, targets, folds_indices, test_fold)

    # normalize data
    X_train = utils.normalize_data(X_train)
    X_test = utils.normalize_data(X_test)

    # Data imputation
    X_train, X_test = impute_data(X_train, X_test, n_neighbors)

    # Feature selection
    X_train, X_test, indices = feature_selection(X_train, X_test, y_train, xgb_k)
    config[f'selected_features_{fold_num}'] = [feature for i, feature in enumerate(labels_vector) if i in indices]

    ### Class balancing ###
    config[f'balance_method_{fold_num}'] = str(balance[fold_num])
    X_train, y_train = balance[fold_num].fit_resample(X_train, y_train)
    class_0 = [zero for i,zero in enumerate(y_train) if y_train[i] == 0]
    class_1 = [one for i,one in enumerate(y_train) if y_train[i] == 1]
    print('class 0:', len(class_0))
    print('class 1:', len(class_1))


    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # model fitting
    estimator1 = estimate_best_model(random_forest('random_forest'), X_train, y_train)
    estimator2 = estimate_best_model(knn('knn'), X_train, y_train)

    estimator_list = [estimator1, estimator2]

    clf, estimator_list, pre_trained_clf = train_model(estimator_list, weight, X_train, y_train)

    if clf is None:
        auroc_vals = []
        aupr_vals = []

    # performance assessment
    roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr = utils.calc_metrics_roc(clf, X_test, y_test, X_train, y_train)
    pr_val, no_skill, lr_recall, lr_precision = utils.calc_metrics_pr(clf, X_test, y_test, X_train, y_train)
    auroc_vals.append([roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr])
    aupr_vals.append([pr_val, no_skill, lr_recall, lr_precision])

    counter += 1

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
        'trained_clf': clf,
        'pre_trained_clf': pre_trained_clf,
        'indices': indices,
        'balance_method': balance[0],
        'exclusion': {
            'model_type': model_type,
            'percentage_removed': percentage_removed,
            'total_removed': total_removed,
            'patient_threshold': patient_threshold
        }
    }


if __name__ == "__main__":
    main()
