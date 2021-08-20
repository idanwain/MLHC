from scipy.stats import ttest_ind
import numpy as np
from sklearn.impute import KNNImputer
import utils
from db_interface_mimic import DbMimic
import os
import pandas as pd
import dataframe_image as dfi

""""
This module runs hypothesis test between group with target = 1 and group with target = 0.
Saves 50 features with most significant difference.
"""

model_type = 'b'
if os.name == 'posix':
    user = 'roye'
else:
    user = 'idan'

boolean_features_path = f'C:/tools/boolean_features_mimic_model_{model_type}_train_data.csv' if user == 'idan' \
    else f'/Users/user/Documents/University/Workshop/boolean_features_mimic_model_{model_type}_train_data.csv'
data_path_mimic = f'C:/tools/external_validation_set_{model_type}_train_data.csv' if user == 'idan' \
    else f'/Users/user/Documents/University/Workshop/external_validation_set_{model_type}_train_data.csv'
folds_path = f'C:/tools/model_{model_type}_folds.csv' if user == 'idan' \
    else f'/Users/user/Documents/University/Workshop/folds_mimic_model_{model_type}.csv'
data_path_eicu = 'C:/tools/model_a_eicu_cohort_training_data.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/model_a_eicu_cohort_training_data.csv'


def get_data_vectors(patient_list):
    _0 = []
    _1 = []
    for patient in patient_list:
        vector = patient.create_vector_for_patient()
        if patient.target == 0:
            _0.append(vector)
        else:
            _1.append(vector)

    return _0, _1


def impute_data(data, n_neighbors):
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    imputer.fit(data)
    data = imputer.transform(data)

    return data


def hypothesis_test():
    db = DbMimic(boolean_features_path,
                 mimic_data_path=data_path_mimic,
                 folds_path=folds_path,
                 eicu_data_path=data_path_eicu if model_type == 'a' else None
                 )
    patient_list = db.create_patient_list()
    labels_vector = utils.create_labels_vector(db, [])

    data_0, data_1 = get_data_vectors(patient_list)
    data = utils.normalize_data(data_0 + data_1)
    data = impute_data(data, 9)
    data_0 = data[:len(data_0)]
    data_1 = data[len(data_0):]

    data_0 = np.array(data_0).transpose()
    data_1 = np.array(data_1).transpose()

    res = []

    for i, (vec_0, vec_1) in enumerate(zip(data_0, data_1)):
        mean_0 = round(float(np.mean(vec_0)), 3)
        mean_1 = round(float(np.mean(vec_1)), 3)
        std_0 = round(float(np.std(vec_0)), 3)
        std_1 = round(float(np.std(vec_1)), 3)
        t_test, p_val = ttest_ind(vec_0, vec_1)
        if p_val < 0.05:
            res.append([labels_vector[i], mean_0, mean_1, std_0, std_1, round(p_val, 3)])

    res.sort(key=lambda row: row[5])
    res = res[:50]
    df = pd.DataFrame(res, columns=['Feature', '0 mean', '1 mean', '0 std', '1 std', 'p-value'])
    df_styled = df.style.background_gradient()  # adding a gradient based on values in cell

    if user == 'idan':
        dfi.export(df_styled, f"C:/tools/objective_{model_type}/hypothesis_test_for_model_{model_type}.png")
    else:
        dfi.export(df_styled, f"/Users/user/Documents/University/Workshop/graphs for milestone 3/hypothesis_test_for_model_{model_type}.png")


if __name__ == '__main__':
    hypothesis_test()
