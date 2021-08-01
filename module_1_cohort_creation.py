import psycopg2
import sql_helper
import os

if os.name == 'posix':
    user = 'roye'
else:
    user = 'idan'

db_conn = psycopg2.connect(
    host="localhost",
    database="mimic",
    user="postgres",
    password="")


def module_1_cohort_creation(file_path, db_conn, model_type):
    db = sql_helper.SqlHelper(db_conn, user)
    db.load_cohort_to_db(file_path, model_type)
    db.create_features_table()
    db.init_boolean_features()
    db.merge_features_and_cohort(user)
    db.close()


if __name__ == '__main__':
    model_type = 'a'
    path = "'C:/tools/model_a_mimic_cohort.csv'" if user == 'idan' else "'/Users/user/Documents/University/Workshop/model_a_mimic_cohort.csv'"
    module_1_cohort_creation(path, db_conn, model_type)
    db_conn.close()
