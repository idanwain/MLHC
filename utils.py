import numpy as np
from matplotlib import pyplot
from sklearn import metrics
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, auc
import one_hot_encoding
from scipy import stats


def get_features_for_removal(threshold: float, patient_list: list, db):
    """
    Returns a list of features to be removed.
    :param threshold: minimum value of appearances a feature should have
    :param patient_list: list of patients
    :param db: DB instance
    :return: list of features
    """
    data_len = len(patient_list)
    labels = db.get_labels()
    labels_dict = {}
    features_to_be_removed = []
    for label in labels:
        labels_dict[label] = 0
    for patient in patient_list:
        for feature in patient.events:
            if len(patient.events[feature]) > 0:
                labels_dict[feature] += (1 / data_len)
    for label in labels:
        if (labels_dict[label] < threshold):
            features_to_be_removed.append(label)
    return features_to_be_removed


def remove_features_by_threshold(threshold: float, patient_list: list, db):
    """
    Removes all features which appear in less than (threshold) of patients
    :param threshold: minimum value of appearances a feature should have
    :param patient_list: list of patients
    :param db: DB instance
    :return: list of patients after removing irrelevant features
    """
    features_to_be_removed = get_features_for_removal(threshold, patient_list, db)
    for patient in patient_list:
        for feature in features_to_be_removed:
            del patient.events[feature]
    return patient_list, features_to_be_removed


def get_top_K_features_xgb(labels_vector, feature_importance: list, k=50):
    indices = []
    list_cpy = feature_importance.copy()
    feature_amount = len(list_cpy)
    if (feature_amount < k):
        log_dict(msg="No %s features in XGB. using %s features instead" % (k, feature_amount))
        print(list_cpy)
        k = feature_amount
    list_cpy.sort(reverse=True)
    for i in range(k):
        indices.append(feature_importance.index(list_cpy[i]))
    # for i in range(k):
    #     index = np.argmax(list_cpy)
    #     indices.append(index)
    #     list_cpy.pop(index)
    # print("Top %s features according to XGB:" % k)
    # for i in indices:
    #     print("Feature: %s, Importance: %s" % (labels_vector[i], feature_importance[i]))
    return indices


def create_vector_of_important_features(data, features):
    """
    Given the top K important features, remove unimportant features.
    :param features:
    :param data: Vectors of features
    :param featurs: top K important features, given by their index
    :return: Set of vectors conatining only relevant features
    """
    new_training_data = []
    for vector in data:
        new_vector = []
        for index in features:
            new_vector.append(vector[index])
        new_training_data.append(new_vector)
    new_training_data = np.asarray(new_training_data)
    return new_training_data


def split_data_by_folds(data, labels, folds, test_fold, removal_factor=0):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    indices_for_removal = []
    counter = 0
    data_len = len(data)
    for i in range(data_len):
        curr_fold = folds[i]
        if curr_fold == test_fold:
            X_test.append(data[i])
            y_test.append(labels[i])
        else:
            X_train.append(data[i])
            y_train.append(labels[i])
    X_train_len = len(X_train)
    for i in range(X_train_len):
        if y_train[i] == 0:
            indices_for_removal.append(i)
            counter += 1
        if counter >= int(X_train_len * removal_factor):
            break

    X_train_removed = []
    y_train_removed = []
    for i in range(X_train_len):
        if i not in indices_for_removal:
            X_train_removed.append(X_train[i])
            y_train_removed.append(y_train[i])

    X_train = X_train_removed
    y_train = y_train_removed
    tot_zero = 0
    tot_one = 1
    for i in range(len(y_train)):
        if y_train[i] == 0:
            tot_zero += 1
        else:
            tot_one += 1
    return X_train, y_train, X_test, y_test


def calc_metrics_roc(clf, X_test, y_test, X_train, y_train, display_plots=False):
    try:
        y_score = clf.predict_proba(X_test)
    except Exception as e:
        calibrator = CalibratedClassifierCV(clf, cv='prefit')
        clf = calibrator.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)

    y_score = y_score[:, 1]
    ns_probs = [0 for _ in range(len(y_test))]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
    lr_auc = roc_auc_score(y_test, y_score)
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_score)

    return lr_auc, ns_fpr, ns_tpr, lr_fpr, lr_tpr


def calc_metrics_pr(clf, x_test, y_test, X_train, y_train):
    try:
        y_score = clf.predict_proba(x_test)
    except Exception as e:
        calibrator = CalibratedClassifierCV(clf, cv='prefit')
        clf = calibrator.fit(X_train, y_train)
        y_score = clf.predict_proba(x_test)
    y_score = y_score[:, 1]
    y_hat = clf.predict(x_test)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_score)
    lr_f1, lr_auc = f1_score(y_test, y_hat), auc(lr_recall, lr_precision)
    positives = len(list(filter(lambda x: x == 1, y_test)))
    no_skill = positives / len(y_test)

    return lr_auc, no_skill, lr_recall, lr_precision


def get_essence_label_vector(labels):
    ret_vector = []
    for label in labels:
        ret_vector.extend([label + "_avg", label + "_max", label + "_min", label + "_latest", label + "_amount"])
    return ret_vector


def log_dict(vals=None, msg=None, log_path="log_file"):
    logging.basicConfig(filename=log_path, filemode='a', level=logging.DEBUG)
    if msg:
        logging.debug(msg)
    if vals:
        logging.debug(str(vals))


def remove_features_by_intersected_list(final_list, patient_list):
    """
    Removes all features that doesn't apear in final_list
    :param final_list: final list of intersected features
    :param patient_list: list of patients
    :return: list of patients after removing irrelevant features
    """
    for patient in patient_list:
        for feature in patient.events.copy():
            if feature not in final_list:
                del patient.events[feature]
    return patient_list


def plot_graphs(auroc_vals, aupr_vals, counter, objective: str):
    for (roc_val, ns_fpr, ns_tpr, lr_fpr, lr_tpr) in auroc_vals:
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label="AUROC= %.3f" % (roc_val))
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()

    avg_auc_str = str(np.average([i[0] for i in auroc_vals]))[2:]
    # pyplot.savefig(f"C:/tools/objective_{objective}/{counter}_auc_{avg_auc_str}.png")
    pyplot.savefig(
        f"/Users/user/Documents/University/Workshop/graphs for milestone 2/{objective}_{counter}_auc_{avg_auc_str}.png")
    pyplot.close()

    for (pr_val, no_skill, lr_recall, lr_precision) in aupr_vals:
        pyplot.plot(lr_recall, lr_precision, marker='.', label="AUPR= %.3f" % (pr_val))
        # axis labels d
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
    avg_aupr_str = str(np.average([i[0] for i in aupr_vals]))[2:]
    # pyplot.savefig(f"C:/tools/objective_{objective}/{counter}_aupr_{avg_aupr_str}.png")
    pyplot.savefig(
        f"/Users/user/Documents/University/Workshop/graphs for milestone 2/{objective}_{counter}_aupr_{avg_aupr_str}.png")
    pyplot.close()


def create_labels_vector(db, removed_features):
    ret_vecotr = []
    essences = ["Average", "Max", "Min", "Latest", "Amount", "STD", "Last 5 average"]
    for label in set(db.get_labels()) - set(removed_features):
        for essence in essences:
            ret_vecotr.extend(label + "_" + essence)
    boolean_features = db.get_distinct_boolean_features()
    boolean_features.sort()
    ret_vecotr.extend(boolean_features)
    ret_vecotr += create_labels_for_categorical_features()
    return ret_vecotr


def create_labels_for_categorical_features():
    return [*one_hot_encoding.GENDER_ENCODING.keys()] + [*one_hot_encoding.INSURANCE_ENCODING.keys()] + \
           [*one_hot_encoding.ETHNICITY_ENCODING.keys()] + ['transfers', 'symptoms']


def normalize_data(data):
    trans = np.array(data).transpose()
    res = []
    for row in trans:
        try:
            z_score = stats.zscore(row, nan_policy='omit')  # maybe should 'omit'
        except Exception as e:
            z_score = row
        res.append(z_score)
    res = np.array(res).transpose()
    return res
