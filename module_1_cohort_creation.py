import sql_helper
import os

try:
    user = os.environ['MLHC_USER']
except KeyError:
    user = 'idan'


def module_1_cohort_creation(file_path, db_conn, model_type):
    db = sql_helper.SqlHelper(db_conn, user)
    db.load_cohort_to_db(file_path, model_type)
    db.create_features_table()
    db.init_boolean_features()
    db.merge_features_and_cohort(user)
