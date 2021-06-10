import copy
import itertools
import random
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import utils
from db_interface_eicu import DbEicu
from db_interface_mimic import DbMimic
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from functools import partial
from imblearn.under_sampling import TomekLinks, ClusterCentroids
from hpsklearn import HyperoptEstimator, svc, any_classifier, any_preprocessing
import numpy as np


user = 'idan'
data_path_eicu = 'C:/tools/feature_eicu_cohort.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/feature_eicu_cohort.csv'
boolean_features_path = 'C:/tools/boolean_features_mimic_model_a.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/boolean_features_mimic_model_a.csv'
extra_features_path = 'C:/tools/extra_features_model_a.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/extra_features_model_a.csv'
data_path_mimic = 'C:/tools/feature_mimic_cohort_model_a.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/feature_mimic_cohort_model_a.csv'
folds_path = 'C:/tools/feature_mimic_cohort_model_a.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/feature_mimic_cohort_model_a.csv'
counter = 1


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def get_intersected_features(db_mimic: DbMimic, db_eicu: DbEicu, pl_mimic, pl_eicu, threshold=0.1,
                             num_of_negatives=1000):
    # patient_list_mimic = db_mimic.create_patient_list(num_of_negatives)
    # patient_list_eicu = db_eicu.create_patient_list()

    patient_list_mimic, removed_features = utils.remove_features_by_threshold(threshold, pl_mimic, db_mimic)
    patient_list_eicu, removed_features = utils.remove_features_by_threshold(threshold, pl_eicu, db_eicu)

    final_list = intersection([*(patient_list_mimic[0].events)], [*(patient_list_eicu[0].events)])

    patient_list_mimic = utils.remove_features_by_intersected_list(final_list, patient_list_mimic)
    patient_list_eicu = utils.remove_features_by_intersected_list(final_list, patient_list_eicu)

    return final_list, patient_list_mimic, patient_list_eicu


def get_target_list(patient_list_mimic, patient_list_eicu):
    targets_mimic = []
    targets_eicu = []
    for patient in patient_list_mimic:
        targets_mimic.append(patient.target)
    for patient in patient_list_eicu:
        targets_eicu.append(patient.target)

    return targets_mimic, targets_eicu


def get_data_from_db(patient_list, labels):
    data = []
    for patient in patient_list:
        vector = patient.create_vector_for_patient(labels, True)
        data.append(vector)
    return data


def fill_missing_data(data, n_neighbors=10):
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    data = (imputer.fit_transform(data))
    return data


def feature_selection(data_mimic, data_eicu, targets_mimic, labels, xgb_k=50):
    model = XGBClassifier()
    model.fit(data_mimic, targets_mimic)
    top_K_xgb = utils.get_top_K_features_xgb(labels, model.feature_importances_.tolist(), k=xgb_k)

    data_mimic = utils.create_vector_of_important_features(data_mimic, top_K_xgb)
    data_eicu = utils.create_vector_of_important_features(data_eicu, top_K_xgb)

    return data_mimic, data_eicu


def build_grid_model(weight):
    # clf1 = DecisionTreeClassifier()
    clf2 = RandomForestClassifier()
    clf3 = KNeighborsClassifier()
    clf = VotingClassifier(estimators=[('rf', clf2), ('knn', clf3)], voting='soft', weights=weight)
    params = {
        'rf__n_estimators': [20, 200],
        'rf__random_state': [0, 5],
        # 'dt__random_state': [0, 5],
        # 'dt__max_depth': [5, 20],
        'knn__n_neighbors': [7, 12],
        'knn__leaf_size': [22, 80]
    }
    return GridSearchCV(estimator=clf, param_grid=params)


def train_model(clf_forest, data_mimic, targets_mimic):
    return clf_forest.fit(data_mimic, targets_mimic)


def model_assesment(clf_forest, data_eicu, targets_eicu, data_mimic, targets_mimic, counter):
    auroc_vals = []
    aupr_vals = []
    roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr = utils.calc_metrics_roc(clf_forest, data_eicu, targets_eicu, data_mimic, targets_mimic)
    pr_val, no_skill, lr_recall, lr_precision = utils.calc_metrics_pr(clf_forest, data_eicu, targets_eicu, data_mimic, targets_mimic)
    auroc_vals.append([roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr])
    aupr_vals.append([pr_val, no_skill, lr_recall, lr_precision])

    return roc_val, pr_val


def objective(params, patient_list_mimic_base, patient_list_eicu_base, db_mimic, db_eicu):
    global counter

    ### Hyperparameters ###
    non = params['non_vals']
    t = params['threshold_vals']
    n = params['kNN_vals']
    clf1_weight = params['clf1_weight']
    clf2_weight = params['clf2_weight']
    weight = [clf1_weight, clf2_weight]
    k = params['XGB_vals']
    config = {
        "Threshold": t,
        "kNN": n,
        "k_XGB": k,
        "non": non,
        "weight": weight
    }
    utils.log_dict(vals=config, msg="Configuration:")

    patient_list_mimic_temp = copy.deepcopy(patient_list_mimic_base)
    random.shuffle(patient_list_mimic_temp)
    neg_num = 0
    patient_list_mimic = []
    for patient in patient_list_mimic_temp:
        if patient.target == 1:
            patient_list_mimic.append(patient)
        elif patient.target == 0 and neg_num < non:
            patient_list_mimic.append(patient)
            neg_num += 1

    patient_list_eicu = copy.deepcopy(patient_list_eicu_base)
    final_labels, patient_list_mimic, patient_list_eicu = get_intersected_features(db_mimic,
                                                                                   db_eicu,
                                                                                   patient_list_mimic,
                                                                                   patient_list_eicu,
                                                                                   threshold=t,
                                                                                   num_of_negatives=non)
    print(final_labels)

    targets_mimic, targets_eicu = get_target_list(patient_list_mimic, patient_list_eicu)

    data_mimic = get_data_from_db(patient_list_mimic, final_labels)
    data_eicu = get_data_from_db(patient_list_eicu, final_labels)

    data_mimic = utils.normalize_data(data_mimic)
    data_eicu = utils.normalize_data(data_eicu)

    data_mimic = fill_missing_data(data_mimic, n_neighbors=n)
    data_eicu = fill_missing_data(data_eicu, n_neighbors=n)

    # final_labels_essence_vector = utils.get_essence_label_vector(final_labels)
    from sklearn.feature_selection import SelectKBest
    # define feature selection
    fs = SelectKBest(k=k)
    # apply feature selection
    data_mimic = fs.fit_transform(data_mimic, targets_mimic)
    top_k_xgb = fs.get_support(indices=True)
    data_eicu = utils.create_vector_of_important_features(data_eicu, top_k_xgb)

    # data_mimic, data_eicu = feature_selection(data_mimic, data_eicu, targets_mimic,
    #                                           final_labels_essence_vector, xgb_k=k)

    # pearson's correlation feature selection for numeric input and numeric output

    under_balancer = TomekLinks()
    data_mimic, targets_mimic = under_balancer.fit_resample(data_mimic, targets_mimic)

    # estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
    #                           algo=tpe.suggest,
    #                           max_evals=20,
    #                           trial_timeout=120)

    # data_mimic = np.array(data_mimic)
    # targets_mimic = np.array(targets_mimic)
    # data_eicu = np.array(data_eicu)
    # targets_eicu = np.array(targets_eicu)
    #
    # auroc_vals = []
    # aupr_vals = []
    # try:
    #     estim.fit(data_mimic, targets_mimic)
    #     print(estim.best_model())
    #     clf = eval(str(estim.best_model()['learner']))
    #     clf = clf.fit(data_eicu, targets_eicu)
    #
    #     ### Performance assement ##
    #     roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr = utils.calc_metrics_roc(clf, data_mimic, targets_mimic, data_eicu, targets_eicu)
    #     pr_val, no_skill, lr_recall, lr_precision = utils.calc_metrics_pr(clf, data_mimic, targets_mimic, data_eicu, targets_eicu)
    #     auroc_vals.append([roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr])
    #     aupr_vals.append([pr_val, no_skill, lr_recall, lr_precision])
    # except Exception as e:
    #     print('exception! aborting..')

    grid = build_grid_model(weight)
    clf_forest = train_model(grid, data_mimic, targets_mimic)
    auroc, aupr = model_assesment(clf_forest, data_eicu, targets_eicu, data_mimic, targets_mimic, counter)

    counter += 1
    results = {
        "AUROC_AVG": auroc,
        "AUPR_AVG": aupr,
    }
    utils.log_dict(vals=results, msg="Run results:")
    return {
        'loss': -1.0 * (aupr + auroc),
        'status': STATUS_OK,
        'metadata': results
    }

def main():
    global counter
    db_mimic = DbMimic(boolean_features_path=boolean_features_path,
                       extra_features_path=extra_features_path,
                       data_path=data_path_mimic,
                       folds_path=folds_path)
    db_eicu = DbEicu(data_path=data_path_eicu)

    patient_list_mimic_base = db_mimic.create_patient_list()
    patient_list_eicu_base = db_eicu.create_patient_list()

    space = {
        'threshold_vals': hp.uniform('thershold_val', 0, 1),
        'kNN_vals': hp.choice('kNN_vals', range(1, 20)),
        'XGB_vals': hp.choice('XGB_vals', range(0, 32)),
        'non_vals': hp.choice('non_vals', range(400, 2000)),
        'clf1_weight': hp.choice('clf1_weight', range(1, 30)),
        'clf2_weight': hp.choice('clf2_weight', range(1, 30))
    }
    objective_func = partial(objective, patient_list_mimic_base=patient_list_mimic_base, patient_list_eicu_base=patient_list_eicu_base, db_mimic=db_mimic, db_eicu=db_eicu)
    trials = Trials()
    best = fmin(fn=objective_func, space=space, algo=tpe.suggest, max_evals=100, trials=trials, return_argmin=False)
    print(best)


if __name__ == "__main__":
    main()
