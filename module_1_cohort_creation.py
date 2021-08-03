import psycopg2
import sql_helper
import os

if os.name == 'posix':
    user = 'roye'
else:
    user = 'idan'


def module_1_cohort_creation(file_path, db_conn, model_type):
    db = sql_helper.SqlHelper(db_conn, user)
    db.load_cohort_to_db(file_path, model_type)
    db.create_features_table()
    db.init_boolean_features()
    db.create_symptoms(model_type)
    output_path = db.merge_features_and_cohort(model_type, user)
    db.close()
    db_conn.close()

    return output_path


if __name__ == '__main__':
    db_conn = psycopg2.connect(
        host="localhost",
        database="mimic",
        user="postgres",
        password="")
    model_type = 'a'
    path = f"'C:/tools/model_{model_type}_mimic_cohort.csv'" if user == 'idan' else f"'/Users/user/Documents/University/Workshop/model_{model_type}_mimic_cohort.csv'"
    module_1_cohort_creation(path, db_conn, model_type)
