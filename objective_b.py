import itertools
from collections import Counter
from threading import Thread

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn.impute import KNNImputer
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import numpy as np
from xgboost import XGBClassifier
from functools import partial
import copy
from db_interface_mimic import DbMimic
import utils
from imblearn.under_sampling import TomekLinks, ClusterCentroids
from imblearn.over_sampling import *
from imblearn.combine import SMOTETomek
from hpsklearn import HyperoptEstimator, svc, any_classifier, any_preprocessing
from numpy import nan

user = 'idan'
boolean_features_path = 'C:/tools/boolean_features_mimic_model_b.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/boolean_features_mimic_model_b.csv'
extra_features_path = 'C:/tools/extra_features_model_b.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/extra_features_model_b.csv'
data_path_mimic = 'C:/tools/feature_mimic_cohort_model_b.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/feature_mimic_cohort_model_b.csv'
folds_path = 'C:/tools/folds_mimic_model_b.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/folds_mimic_model_b.csv'


def run_fold(index, auroc_vals, aupr_vals, data, targets, folds_indices, test_fold):
    ### Data split ###
    X_train, y_train, X_test, y_test = utils.split_data_by_folds(data, targets, folds_indices, test_fold)

    ### Feature selection ###
    # model = XGBClassifier(use_label_encoder=False)
    # model.fit(np.asarray(X_train), y_train)
    # top_K_xgb = utils.get_top_K_features_xgb(labels_vector, model.feature_importances_.tolist(), k=xgb_k)
    # X_train = utils.create_vector_of_important_features(X_train, top_K_xgb)
    # X_test = utils.create_vector_of_important_features(X_test, top_K_xgb)

    ### Class balancing ###
    # over_balancer = BorderlineSMOTE()
    # X_train, y_train = over_balancer.fit_resample(X_train, y_train)
    under_balancer = TomekLinks()
    X_train, y_train = under_balancer.fit_resample(X_train, y_train)

    ### Model fitting ###
    # clf1 = RandomForestClassifier()
    # clf2 = KNeighborsClassifier()
    # clf3 = DecisionTreeClassifier()
    estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                              algo=tpe.suggest,
                              max_evals=10,
                              trial_timeout=120)

    # clf = VotingClassifier(estimators=[('clf1', clf1)], voting='soft') # , weights=weight
    # clf = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2), ('clf3', clf3)], voting='soft', weights=weight)
    # params = {
    #     'rf__n_estimators': [50, 175],
    #     # 'rf__max_depth': [5, 100],
    #     'rf__random_state': [0, 3],
    #     # 'dt__random_state': [0, 5],
    #     # 'dt__max_depth': [5, 20],
    #     'knn__n_neighbors': [1, 20],
    #     'knn__leaf_size': [15, 40]
    # }
    # grid = GridSearchCV(estimator=clf, param_grid=params)
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
        auroc_vals[index] = [roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr]
        aupr_vals[index] = [pr_val, no_skill, lr_recall, lr_precision]

    except Exception as e:
        auroc_vals[index] = None
        aupr_vals[index] = None


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
        # 'XGB_vals': hp.choice('XGB_vals', range(30, 62)),
        # 'clf1_weight_vals': hp.choice('clf1_weight_vals', range(1, 30)),
        # 'clf2_weight_vals': hp.choice('clf2_weight_vals', range(1, 30)),
        # 'clf3_weight_vals': hp.choice('clf3_weight_vals', range(1, 30))
        # 'removal_factor_vals': hp.uniform('removal_factor_vals', 0, 0.2)
    }
    objective_func = partial(objective, patient_list_base=patient_list_base, db=db, folds=folds)
    trials = Trials()
    best = fmin(fn=objective_func, space=space, algo=tpe.suggest, max_evals=50, trials=trials, return_argmin=False)
    print(best)


def objective(params, patient_list_base, db, folds):
    global counter
    counter = 0
    data = []
    targets = []
    folds_indices = []
    auroc_vals = [None] * 5
    aupr_vals = [None] * 5
    patient_list = copy.deepcopy(patient_list_base)

    ### Hyperparameters ###
    threshold = params['threshold_vals']  # Minimum appearance for feature to be included
    n_neighbors = params['kNN_vals']  # Neighbors amount for kNN
    # xgb_k = params['XGB_vals']  # Amount of features to return by XGB
    # removal_factor = params['removal_factor_vals']  # How many negative samples we remove from the training set.
    # weight = list(params['weight_vals'])
    # clf1_weight = params['clf1_weight_vals']
    # clf2_weight = params['clf2_weight_vals']
    # clf3_weight = params['clf3_weight_vals']
    # weight = [clf1_weight, clf2_weight, clf3_weight]
    config = {
        "Threshold": threshold,
        "kNN": n_neighbors,
        # "k_XGB": xgb_k,
        # "Removal factor": removal_factor,
        # 'weights': weight
    }
    utils.log_dict(vals=config, msg="Configuration:")

    patient_list, removed_features = utils.remove_features_by_threshold(threshold, patient_list, db)
    for patient in patient_list:
        targets.append(patient.target)
        for fold in folds:
            if str(patient.hadm_id) in (folds[fold]):
                folds_indices.append(fold)

    # labels_vector = utils.create_labels_vector(db, removed_features)
    for patient in patient_list:
        vector = patient.create_vector_for_patient()
        data.append(vector)

    ### Normalize data ###
    data = utils.normalize_data(data)

    ### Data imputation ###
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    data = (imputer.fit_transform(data))

    threads = []
    for i, test_fold in enumerate(folds):
        threads.append(Thread(target=run_fold, args=(i, auroc_vals, aupr_vals, data, targets, folds_indices, test_fold,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()


    # auroc_vals.append([roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr])
    # aupr_vals.append([pr_val, no_skill, lr_recall, lr_precision])

    # utils.plot_graphs(auroc_vals, aupr_vals, counter, 'b')

    counter += 1

    if None in auroc_vals or None in aupr_vals:
        return {
            'loss': 0,
            'status': STATUS_OK,
        }

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
