import numpy as np
from matplotlib import pyplot
from sklearn import metrics
import logging
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, auc


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
        for feature in patient.events.copy():
            if feature in features_to_be_removed:
                del patient.events[feature]
    return patient_list


def get_top_K_features_xgb(labels_vector, feature_importance: list, k=50):
    indices = []
    list_cpy = feature_importance.copy()
    feature_amount = len(list_cpy)
    if (feature_amount < k):
        log_dict(msg="No %s features in XGB. using %s features instead" % (k, feature_amount))
        print(list_cpy)
        k = feature_amount
    for i in range(k):
        index = np.argmax(list_cpy)
        indices.append(index)
        list_cpy.pop(index)

    # Print list of features, can be removed
    # print("Top %s features according to XGB:" % k)
    # for i in indices:
    #     print("Feature: %s, Importance: %s" % (labels_vector[i], feature_importance[i]))
    return indices


def create_vector_of_important_features(data, features: list):
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


def split_data(data, labels, ratio):
    """
    Splits the data into Traning data and test data. Same for the labels.
    The split is currently done hard coded and set to 70% of the data.

    :param data: Array of vectors
    :param labels: Binary vector
    :return: X_train,y_train,X_test,y_test - traning and test data and labels.
    """

    # Doing this so i can randomize the data without losing relation to labels
    joined_data = []
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    pos = []
    neg = []
    data_len = len(data)
    for i in range(data_len):
        joined_data.append((data[i], labels[i]))
    np.random.shuffle(joined_data)
    for vector in joined_data:
        if (vector[1] == 0):
            neg.append(vector)
        else:
            pos.append(vector)
    pos_split_index = int(len(pos) * 0.7)
    neg_split_index = int((len(neg) * 0.7) / ratio)
    train = neg[:neg_split_index] + pos[:pos_split_index]
    test = neg[neg_split_index:] + pos[pos_split_index:]
    # train = joined_data[:int(len(joined_data)*0.7)]
    # test = joined_data[int(len(joined_data)*0.7):]
    for vector in train:
        X_train.append(list(vector[0]))
        y_train.append(vector[1])
    for vector in test:
        X_test.append(list(vector[0]))
        y_test.append(vector[1])
    return X_train, y_train, X_test, y_test


def split_data_by_folds(data, labels, folds, test_fold, removal_factor=1):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    indices_for_removal = []
    counter = 0
    data_len = len(data)
    for i in range(data_len):
        curr_fold = folds[i]
        if (curr_fold == test_fold):
            X_test.append(data[i])
            y_test.append(labels[i])
        else:
            X_train.append(data[i])
            y_train.append(labels[i])
    X_train_len = len(X_train)
    # print("Y len: %s. X len: %s" %(len(y_train),X_train_len))
    for i in range(X_train_len):
        if y_train[i] == 0:
            indices_for_removal.append(i)
            counter += 1
        if (counter == int(X_train_len * removal_factor)):
            break;

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
        if (y_train[i] == 0):
            tot_zero += 1
        else:
            tot_one += 1
    # print("Ratio: %s"%(tot_zero/tot_one))
    # print("Y len: %s. X len: %s" %(len(y_train),len(X_train)))
    return X_train, y_train, X_test, y_test


def calc_error(clf, X_test, y_test):
    tot = 0
    for i in range(len(X_test)):
        val = clf.predict([X_test[i]])
        if (val != y_test[i]):
            tot += 1
    print(1 - (tot / len(X_test)))
    return (1 - (tot / len(X_test)))


def calc_metrics(clf, X_test, y_test, display_plots=False):
    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    ns_probs = [0 for _ in range(len(y_test))]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc1 = roc_auc_score(y_test, y_score)
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc1))
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_score)
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

    y_score = clf.predict_proba(X_test)
    y_score = y_score[:, 1]
    yhat = clf.predict(X_test)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_score)
    lr_f1, lr_auc2 = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc2))
    # plot the precision-recall curves
    positives = len(list(filter(lambda x: x == 1, y_test)))
    no_skill = positives / len(y_test)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

    return lr_auc1, lr_auc2


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
