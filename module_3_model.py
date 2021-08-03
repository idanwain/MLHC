import pickle
import pandas as pd
import os


path_to_model = "model"
if os.name == 'posix':
    user = 'roye'
else:
    user = 'idan'

def module_3_model(processed_external_validation_set_path, model_type):
    path_to_risk_score = 'C:/tools/model_' + model_type + '_mimic_cohort_risk_score_group_8.csv' if user == 'idan' \
        else '/Users/user/Documents/University/Workshop/model_' + model_type + '_mimic_cohort_risk_score_group_8.csv'

    final_df = {'identifier': [], 'risk_score': []}

    ### Load data from csv ###
    df = pd.read_csv(processed_external_validation_set_path)
    for row in df.iterrows():
        final_df['identifier'].append(row[1]['identifier'])
    df.drop('identifier')

    ### Load model and perform prediciton ###
    data = df.to_numpy()
    with open(path_to_model + '_' + model_type , 'rb') as file:
        clf = pickle.load(file)
    result = clf.predict_proba(data)

    ### Save results to csv ###
    final_df['risk_score'].extend(result)
    res = pd.DataFrame.from_dict(final_df)
    res.to_csv(path_to_risk_score)
