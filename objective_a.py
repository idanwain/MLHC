import itertools
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, atpe
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import numpy as np
from xgboost import XGBClassifier
from functools import partial
import copy
from db_interface_mimic import DbMimic
import utils
from imblearn.under_sampling import TomekLinks, ClusterCentroids, RandomUnderSampler
from imblearn.over_sampling import *
from imblearn.combine import SMOTETomek
from hpsklearn import HyperoptEstimator, svc, any_classifier, any_preprocessing
from numpy import nan
import os

if os.name == 'posix':
    user = 'roye'
else:
    user = 'idan'
boolean_features_path = 'C:/tools/boolean_features_mimic_model_a.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/boolean_features_mimic_model_a.csv'
extra_features_path = 'C:/tools/extra_features_model_a.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/extra_features_model_a.csv'
data_path_mimic = 'C:/tools/feature_mimic_cohort_model_a.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/feature_mimic_cohort_model_a.csv'
folds_path = 'C:/tools/folds.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/folds_mimic_model_a.csv'

counter = 1


def main():
    db = DbMimic(boolean_features_path,
                 extra_features_path,
                 data_path=data_path_mimic,
                 folds_path=folds_path)

    folds = db.get_folds()
    patient_list_base = db.create_patient_list()
    space = {
        'threshold_vals': hp.uniform('thershold_val', 0, 1),
        'kNN_vals': hp.choice('kNN_vals', range(1, 20)),
        'XGB_vals': hp.choice('XGB_vals', range(30, 62))
    }
    objective_func = partial(objective,patient_list_base=patient_list_base,db=db,folds=folds)
    trials = Trials()
    best = fmin(fn=objective_func, space=space, algo=atpe.suggest, max_evals=5, trials=trials, return_argmin=False)
    print(best)


def objective(params,patient_list_base,db,folds):
    global counter
    data = []
    targets = []
    folds_indices = []
    auroc_vals = []
    aupr_vals = []
    selected_features = []
    patient_list = copy.deepcopy(patient_list_base)

    ### Hyperparameters ###
    threshold = params['threshold_vals']  # Minimum appearance for feature to be included
    n_neighbors = params['kNN_vals']  # Neighbors amount for kNN
    xgb_k = params['XGB_vals']  # Amount of features to return by XGB
    config = {
        "Threshold": threshold,
        "kNN": n_neighbors,
        "K_best": xgb_k,
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

    data = utils.normalize_data(data)

    for test_fold in folds:
        ### Data split ###
        X_train, y_train, X_test, y_test = utils.split_data_by_folds(data, targets, folds_indices, test_fold)

        ### Data imputation ###
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)

        ### Feature selection ###
        selector = SelectKBest(k=xgb_k)
        X_train = selector.fit_transform(X_train,y_train)
        indices = selector.get_support(indices=True)
        selected_features.append([feature for i, feature in enumerate(labels_vector) if i in indices])
        X_test = utils.create_vector_of_important_features(X_test, indices)


        ### Class balancing ###
        # over_balancer = BorderlineSMOTE()
        # X_train, y_train = over_balancer.fit_resample(X_train, y_train)

        under_balancer = RandomUnderSampler()
        X_train, y_train = under_balancer.fit_resample(X_train,y_train)
        # under_balancer = TomekLinks()
        # X_train, y_train = under_balancer.fit_resample(X_train, y_train)
        # combined_balancer = SMOTETomek()
        # X_train, y_train = combined_balancer.fit_resample(X_train,y_train)


        ### Model fitting ###
        estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                                  algo=tpe.suggest,
                                  max_evals=10,
                                  trial_timeout=60)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        try:
            estim.fit(X_train, y_train)
            print(estim.best_model())
            clf = eval(str(estim.best_model()['learner']))
            clf = clf.fit(X_train, y_train)

            ### Performance assement ##
            roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr = utils.calc_metrics_roc(clf, X_test, y_test, X_train, y_train)
            pr_val, no_skill, lr_recall, lr_precision = utils.calc_metrics_pr(clf, X_test, y_test, X_train, y_train)
        except Exception as e:
            auroc_vals = []
            aupr_vals = []
            utils.log_dict(msg=f"ERROR: {e}")
            break

        auroc_vals.append([roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr])
        aupr_vals.append([pr_val, no_skill, lr_recall, lr_precision])

    utils.plot_graphs(auroc_vals, aupr_vals, counter, 'a')
    config['selected_features'] = selected_features
    utils.save_conf_file(config, counter, 'a')

    counter += 1
    results = {"AUROC_AVG": np.average([i[0] for i in auroc_vals]),
                         "AUPR_AVG": np.average([i[0] for i in aupr_vals]),
                         "AUROC_STD": np.std([i[0] for i in auroc_vals]),
                         "AUPR_STD": np.std([i[0] for i in aupr_vals])}
    utils.log_dict(vals=results, msg="Run results:")
    return {
        'loss': -1.0 * np.average([i[0] for i in auroc_vals]),
        'status': STATUS_OK,
        'metadata': results
    }


if __name__ == "__main__":
    main()
