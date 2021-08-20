from db_interface_mimic import DbMimic
import utils
from sklearn.impute import KNNImputer
import pandas as pd
import pickle
import os

if os.name == 'posix':
    user = 'roye'
else:
    user = 'idan'

processed_external_validation_set_path = 'C:/tools/processed_external_validation_set.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/processed_external_validation_set.csv'
data_path_eicu = 'C:/tools/model_a_eicu_cohort.csv' if user == 'idan'\
    else '/Users/user/Documents/University/Workshop/model_a_eicu_cohort.csv'


def module_2_preprocessing(external_validation_set_path, model_type):
    indices_file_path = 'indices_' + model_type
    optimal_values_path = 'optimal_values_' + model_type
    exclusion_path = 'exclusion_criteria_' + model_type
    threshold, n_neighbors = load_optimal_values_from_disk(optimal_values_path)
    exclusion_data = load_exclusion_from_disk(exclusion_path)
    data = []
    ids = {'identifier': []}

    boolean_feature_path = f'C:/tools/boolean_features_mimic_model_{model_type}.csv' if user == 'idan' \
        else f'/Users/user/Documents/University/Workshop/boolean_features_mimic_model_{model_type}.csv'
    db = DbMimic(boolean_features_path=boolean_feature_path,
                 mimic_data_path=external_validation_set_path,
                 folds_path=None,
                 eicu_data_path=None
                 )
    patient_list_base = db.create_patient_list()

    patient_list, percentage_removed, total_removed = utils.remove_patients_by_thershold(patient_list_base,
                                                                                         exclusion_data[
                                                                                             'patient_threshold'])

    ### Removing rare features ###
    patient_list, removed_features = utils.remove_features_by_threshold(threshold, patient_list, db)
    for patient in patient_list:
        vector = patient.create_vector_for_patient()
        data.append(vector)
        ids['identifier'].append(patient.get_identifier())

    ### Normalize data ###
    data = utils.normalize_data(data)

    ### Impute data ###
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    imputer.fit(data)
    data = imputer.transform(data)

    ### Feature selection ###
    indices = load_indices_from_disk(indices_file_path)
    data = utils.create_vector_of_important_features(data, indices)

    ### Save files ###
    df = pd.DataFrame(data)
    df['identifier'] = ids['identifier']
    df.to_csv(processed_external_validation_set_path)
    write_exclusion_file(exclusion_data, percentage_removed, total_removed, model_type)
    return processed_external_validation_set_path


def load_indices_from_disk(indices_file_path):
    with open(indices_file_path, 'rb') as file:
        indices = pickle.load(file)
    return indices


def load_optimal_values_from_disk(optimal_values_path):
    with open(optimal_values_path, 'rb') as file:
        data = pickle.load(file)
    return data['feature_threshold'], data['kNN_vals']


def load_exclusion_from_disk(exclusion_path):
    with open(exclusion_path, 'rb') as file:
        data = pickle.load(file)
    return data


def write_exclusion_file(exclusion_data, percentage_removed, total_removed, model_type):
    file_content = f"Exclusion criteria are:\n1. All patients with less than {exclusion_data['patient_threshold'] * 100}% " \
                   "out of total features were removed."
    file_content += f"\nOn model type {model_type}: {exclusion_data['total_removed']} were removed ({exclusion_data['percentage_removed'] * 100}" \
                    "% of the cohort)"
    file_content += f"\n{total_removed} patients were removed from the external validation set ({percentage_removed}%)"

    with open("cohort_exclusion.txt", 'w') as file:
        file.write(file_content)
