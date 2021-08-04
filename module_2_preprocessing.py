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

processed_external_validation_set_path = 'C:/tools/processed_external_validation_set_path.csv' if user == 'idan' \
    else '/Users/user/Documents/University/Workshop/processed_external_validation_set_path.csv'
def module_2_preprocessing(external_validation_set_path, model_type):
    indices_file_path = 'indices_' + model_type
    optimal_values_path = 'optimal_values_' + model_type
    threshold , n_neighbors = load_optimal_values_from_disk(optimal_values_path)
    data = []
    ids = {'identifier':[]}
    db = DbMimic(f'/Users/user/Documents/University/Workshop/boolean_features_mimic_model_{model_type}.csv',
                 # external_validation_set_path,
                 data_path=external_validation_set_path
                 )
    patient_list_base = db.create_patient_list()

    ### Removing rare feautres ###
    patient_list, removed_features = utils.remove_features_by_threshold(threshold, patient_list_base, db)
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
    data = utils.create_vector_of_important_features(data,indices)

    ### Save files ###
    df = pd.DataFrame(data)
    df['identifier'] = ids['identifier']
    df.to_csv(processed_external_validation_set_path)
    return processed_external_validation_set_path

def load_indices_from_disk(indices_file_path):
    with open(indices_file_path,'rb') as file:
        indices = pickle.load(file)
    return indices

def load_optimal_values_from_disk(optimal_values_path):
    with open(optimal_values_path,'rb') as file:
        data = pickle.load(file)
    return data['threshold_vals'], data['kNN_vals']