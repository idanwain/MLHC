import psycopg2
import sql_helper
import os

if os.name == 'posix':
    user = 'roye'
else:
    user = 'idan'

def module_1_cohort_creation(file_path, db_conn, model_type,training=False):
    db = sql_helper.SqlHelper(db_conn, model_type, user, training)
    db.load_cohort_to_db(file_path)
    db.create_features_table()
    db.init_boolean_features()
    db.create_symptoms()
    db.merge_features_and_cohort()
    db.close()
    db_conn.close()

    return f"C:/tools/external_validation_set_{model_type}.csv" if user == 'idan' \
        else f"/Users/user/Documents/University/Workshop/external_validation_set_{model_type}.csv"


def create_cohort_training_data():
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
    model_type = 'b'
    path = f"'C:/tools/model_{model_type}_mimic_cohort.csv'" if user == 'idan' else f"'/Users/user/Documents/University/Workshop/model_{model_type}_mimic_cohort.csv'"
    module_1_cohort_creation(path, db_conn, model_type,training=True)
