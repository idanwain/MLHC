import copy
import random
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import utils
from db_interface_eicu import DbEicu
from db_interface_mimic import DbMimic
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, rand, atpe
from functools import partial
from imblearn.under_sampling import TomekLinks
from hpsklearn import HyperoptEstimator, random_forest, knn, ada_boost
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


def build_patients_list(patient_list_mimic_base, patient_list_eicu_base, non):
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

    return patient_list_mimic, patient_list_eicu


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def get_intersected_features(db_mimic: DbMimic, db_eicu: DbEicu, pl_mimic, pl_eicu, threshold=0.1):
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


def feature_selection(data_mimic, data_eicu, targets_mimic, k):
    fs = SelectKBest(k=k)
    data_mimic = fs.fit_transform(data_mimic, targets_mimic)
    top_k_xgb = fs.get_support(indices=True)
    data_eicu = utils.create_vector_of_important_features(data_eicu, top_k_xgb)

    return data_mimic, data_eicu


def build_grid_model(clfs, weight):
    # clf1 = RandomForestClassifier()
    # clf2 = KNeighborsClassifier()
    # clf3 = AdaBoostClassifier()
    # estimators = [('clf1', clf1), ('clf2', clf2), ('clf3', clf3)]
    estimators = [(f'clf{i+1}', clf) for i, clf in enumerate(clfs)]
    clf = VotingClassifier(estimators=estimators, voting='soft', weights=weight)
    return clf
    # params = {
    #     'rf__n_estimators': [20, 200],
    #     'rf__random_state': [0, 5],
    #     'knn__n_neighbors': [1, 20],
    #     'knn__leaf_size': [22, 80],
    #     'ab__n_estimators': [20, 150]
    # }
    # return GridSearchCV(estimator=clf, param_grid=params)


def train_model(clf_forest, data_mimic, targets_mimic):
    return clf_forest.fit(data_mimic, targets_mimic)


def model_assessment(clf_forest, data_eicu, targets_eicu, data_mimic, targets_mimic):
    auroc_vals = []
    aupr_vals = []
    roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr = utils.calc_metrics_roc(clf_forest, data_eicu, targets_eicu, data_mimic,
                                                                     targets_mimic)
    pr_val, no_skill, lr_recall, lr_precision = utils.calc_metrics_pr(clf_forest, data_eicu, targets_eicu, data_mimic,
                                                                      targets_mimic)
    auroc_vals.append([roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr])
    aupr_vals.append([pr_val, no_skill, lr_recall, lr_precision])

    return roc_val, pr_val, auroc_vals, aupr_vals


def estimate_best_model(clf, data_mimic, targets_mimic):
    estimator = HyperoptEstimator(classifier=clf,
                                  algo=tpe.suggest,
                                  max_evals=5,
                                  trial_timeout=120)
    try:
        estimator.fit(data_mimic, targets_mimic)
        res = eval(str(estimator.best_model()['learner']))
    except Exception as e:
        res = RandomForestClassifier()
    return res


def objective(params, patient_list_mimic_base, patient_list_eicu_base, db_mimic, db_eicu):
    global counter

    # hyper-parameters
    non = params['non_vals']
    t = params['threshold_vals']
    n = params['kNN_vals']
    clf1_weight = params['clf1_weight']
    clf2_weight = params['clf2_weight']
    # clf3_weight = params['clf3_weight']
    weight = [clf1_weight, clf2_weight]
    k = params['XGB_vals']
    over_balance = params['over_balance']

    # log configuration
    config = {
        "Threshold": t,
        "kNN": n,
        "k_XGB": k,
        "non": non,
        "weight": weight,
        "over_balance": over_balance,
        "counter": counter
    }
    utils.log_dict(vals=config, msg="Configuration:")

    # build patients lists
    patient_list_mimic, patient_list_eicu = build_patients_list(patient_list_mimic_base, patient_list_eicu_base, non)

    # intersect features between dbs
    final_labels, patient_list_mimic, patient_list_eicu = get_intersected_features(db_mimic,
                                                                                   db_eicu,
                                                                                   patient_list_mimic,
                                                                                   patient_list_eicu,
                                                                                   threshold=t)
    # get targets
    targets_mimic, targets_eicu = get_target_list(patient_list_mimic, patient_list_eicu)

    # extract data from db
    data_mimic = get_data_from_db(patient_list_mimic, final_labels)
    data_eicu = get_data_from_db(patient_list_eicu, final_labels)

    # normalize data
    data_mimic = utils.normalize_data(data_mimic)
    data_eicu = utils.normalize_data(data_eicu)

    # fill missing data
    data_mimic = fill_missing_data(data_mimic, n_neighbors=n)
    data_eicu = fill_missing_data(data_eicu, n_neighbors=n)

    # feature selection
    data_mimic, data_eicu = feature_selection(data_mimic, data_eicu, targets_mimic, k)

    # balance data
    if over_balance == 0:
        over_balancer = BorderlineSMOTE()
        data_mimic, targets_mimic = over_balancer.fit_resample(data_mimic, targets_mimic)
    elif over_balance == 1:
        under_balancer = TomekLinks()
        data_mimic, targets_mimic = under_balancer.fit_resample(data_mimic, targets_mimic)

    # weight = 1
    # adjust data for estimator
    data_mimic = np.array(data_mimic)
    targets_mimic = np.array(targets_mimic)
    data_eicu = np.array(data_eicu)
    targets_eicu = np.array(targets_eicu)

    # estimate best model
    clf1 = estimate_best_model(random_forest('my_name.random_forest'), data_mimic, targets_mimic)
    clf2 = estimate_best_model(knn('my_name.knn'), data_mimic, targets_mimic)
    # clf3 = estimate_best_model(ada_boost('my_name.ada_boost'), data_mimic, targets_mimic)
    clfs = [clf1, clf2]

    # fit model
    grid = build_grid_model(clfs, weight)
    clf_forest = train_model(grid, data_mimic, targets_mimic)

    # model assessment
    auroc, aupr, auroc_vals, aupr_vals = model_assessment(clf_forest, data_eicu, targets_eicu, data_mimic,
                                                          targets_mimic)

    # plot graph
    utils.plot_graphs(auroc_vals, aupr_vals, counter, 'c')

    counter += 1
    results = {
        "AUROC_AVG": auroc,
        "AUPR_AVG": aupr,
    }
    utils.log_dict(vals=results, msg="Run results:")
    return {
        'loss': -1.0 * aupr,
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
        'kNN_vals': hp.choice('kNN_vals', range(1, 100)),
        'XGB_vals': hp.choice('XGB_vals', range(1, 60)),
        'non_vals': hp.choice('non_vals', range(400, 2500)),
        'clf1_weight': hp.choice('clf1_weight', range(1, 30)),
        'clf2_weight': hp.choice('clf2_weight', range(1, 30)),
        # 'clf3_weight': hp.choice('clf3_weight', range(1, 30)),
        'over_balance': hp.choice('over_balance', [0, 1, 2])
    }
    objective_func = partial(
        objective,
        patient_list_mimic_base=patient_list_mimic_base,
        patient_list_eicu_base=patient_list_eicu_base,
        db_mimic=db_mimic,
        db_eicu=db_eicu
    )
    trials = Trials()
    best = fmin(
        fn=objective_func,
        space=space,
        algo=atpe.suggest,
        max_evals=1000,
        trials=trials,
        return_argmin=False
    )
    print(best)


if __name__ == "__main__":
    main()
