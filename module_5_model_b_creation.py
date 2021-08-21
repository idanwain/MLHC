import objective_b
import os

if os.name == 'posix':
    user = 'roye'
else:
    user = 'idan'


def module_5_model_b_creation(model_type, model_b_mimic_cohort_csv):
    objective_b.main(given_model_type=model_type, given_mimic_data_path=model_b_mimic_cohort_csv)


def main():
    model_type = 'b'
    data_path_mimic = f'C:/tools/external_validation_set_{model_type}_train_data.csv' if user == 'idan' \
        else f'/Users/user/Documents/University/Workshop/external_validation_set_{model_type}_train_data.csv'
    module_5_model_b_creation(model_type, data_path_mimic)
