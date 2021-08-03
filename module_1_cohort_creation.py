import psycopg2
import sql_helper
import os

if os.name == 'posix':
    user = 'roye'
else:
    user = 'idan'

training = False


def module_1_cohort_creation(file_path, db_conn, model_type):
    db = sql_helper.SqlHelper(db_conn, model_type, user, training)
    print('1')
    db.load_cohort_to_db(file_path)
    print('2')
    db.create_features_table()
    print('3')
    db.init_boolean_features()
    print('4')
    db.create_symptoms()
    print('5')
    db.merge_features_and_cohort()
    print('6')
    db.close()
    db_conn.close()

    return f"C:/tools/external_validation_set_{model_type}.csv" if user == 'idan' \
        else f"/Users/user/Documents/University/Workshop/external_validation_set_{model_type}.csv"


if __name__ == '__main__':
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
    training = True
    path = f"'C:/tools/model_{model_type}_mimic_cohort.csv'" if user == 'idan' else f"'/Users/user/Documents/University/Workshop/model_{model_type}_mimic_cohort.csv'"
    module_1_cohort_creation(path, db_conn, model_type)
