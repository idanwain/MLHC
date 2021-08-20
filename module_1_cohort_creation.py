import psycopg2
import sql_helper
import os

if os.name == 'posix':
    user = 'roye'
else:
    user = 'idan'

"""
This module collects data of given cohort from DB.
Returns a CSV of external validation set that contains
"""


def module_1_cohort_creation(file_path, db_conn, model_type, training=False):
    db = sql_helper.SqlHelper(db_conn, model_type, user, training)
    db.load_cohort_to_db(file_path)
    db.create_features_table()
    db.init_boolean_features()
    db.create_drug_table()
    db.create_symptoms()
    db.merge_features_and_cohort()
    db.close()
    db_conn.close()

    return f"C:/tools/external_validation_set_{model_type}.csv" if user == 'idan' \
        else f"/Users/user/Documents/University/Workshop/external_validation_set_{model_type}.csv"


def eicu_cohort_creation():
    if user == 'idan':
        db_conn = psycopg2.connect(
            host="localhost",
            database="eicu",
            user="postgres",
            password="",
            options="--search_path=eicu"
        )
    else:  # TODO: set configuration for Roye
        db_conn = psycopg2.connect(
            host="localhost",
            database="eicu",
            user="eicuuser",
            password="",
            options="--search_path=eicu"
        )

    db = sql_helper.SqlHelper(db_conn, 'a', user, True)
    file_path = f"'C:/tools/model_a_eicu_cohort.csv'" if user == 'idan' else f"'/Users/user/Documents/University/Workshop/model_a_eicu_cohort.csv'"
    db.load_eicu_cohort_to_db(file_path)
    db.create_eicu_to_mimic_mapping()
    db.create_eicu_features_table()
    db.close()
    db_conn.close()


def create_cohort_training_data(model_type):
    print("Creating cohort training data")
    if user == 'idan':
        db_conn = psycopg2.connect(
            host="localhost",
            database="mimic",
            user="postgres",
            password="")
    else:
        db_conn = psycopg2.connect(
            host="localhost",
            database="mimic",
            user="mimicuser",
            password="",
            options="--search_path=mimiciii"
        )
    mimic_path = f"'C:/tools/model_{model_type}_mimic_cohort.csv'" if user == 'idan' else f"'/Users/user/Documents/University/Workshop/model_{model_type}_mimic_cohort.csv'"
    module_1_cohort_creation(mimic_path, db_conn, model_type, training=True)
    if model_type == 'a':
        eicu_cohort_creation()
