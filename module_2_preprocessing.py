from db_interface_mimic import DbMimic
import utils
from sklearn.impute import KNNImputer
import pandas as pd

indices_file_path = './indices'
processed_external_validation_set_path = './processed_external_validation_set_path.csv'
def module_2_preprocessing(external_validation_set_path, model_type):
    ### These values should be replaced with optimal values ###
    threshold = 0.5
    n_neighbors = 5
    data = []
    db = DbMimic(external_validation_set_path,
                 external_validation_set_path,
                 data_path=external_validation_set_path,
                 folds_path='')
    patient_list_base = db.create_patient_list()

    ### Removing rare feautres ###
    patient_list, removed_features = utils.remove_features_by_threshold(threshold, patient_list_base, db)
    for patient in patient_list:
        vector = patient.create_vector_for_patient()
        data.append(vector)

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
    df.to_csv(processed_external_validation_set_path)

def load_indices_from_disk(indices_file_path):
    pass

