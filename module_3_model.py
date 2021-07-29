import pickle
import pandas as pd

path_to_model = "./model"

def module_3_model(processed_external_validation_set_path,model_type):
    df = pd.read_csv(processed_external_validation_set_path)
    data = df.to_numpy()
    clf = pickle.load(path_to_model)
    result = clf.predict_proba(data)