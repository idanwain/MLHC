from typing import List
import utils


class SqlHelper:
    def __init__(self, db_connection, username):
        self.cursor = db_connection.cursor()
        self.user = username

    def execute_query(self, query: str, params: List = None):
        try:
            if params is not None:
                execute_result = self.cursor.execute(query, params)
            else:
                execute_result = self.cursor.execute(query)
        except Exception as e:
            utils.log_dict(msg=f'DB error: {query}, Exception: {e}')
            raise

        result = []
        column = []
        if execute_result is not None:
            if not execute_result.description:
                return []

            columns = [column[0] for column in execute_result.description]

            if len(columns) == 1:
                result.append({columns[0]: []})
                column = result[0].get(columns[0])

            for row in execute_result.fetchall():
                if len(columns) == 1:
                    column.append(row[0])
                else:
                    result.append(dict(zip(columns, row)))

            if len(result) == 1 and len(result[0].keys()) == 1 and not result[0][list(result[0].keys())[0]]:
                result = []

        return result

    def load_cohort_to_db(self, file_path, model_type):
        query = f"""DROP TABLE IF EXISTS model_{model_type}_mimic_cohort;
                    CREATE TABLE model_{model_type}_mimic_cohort (
                      identifier VARCHAR(50),
                      subject_id VARCHAR(50),
                      hadm_id VARCHAR(50),
                      admittime TIMESTAMP,
                      icu_time TIMESTAMP,
                      target_time TIMESTAMP,
                      target VARCHAR(50)
                    );
                    COPY model_{model_type}_mimic_cohort 
                    FROM {file_path}
                    DELIMITER ','
                    CSV HEADER;"""

        self.execute_query(query)

    def create_features_table(self):
        query = f"""DROP TABLE IF EXISTS cohort_relevant_features;
                    create table cohort_relevant_features(
                        item_id INT,
                        _table TEXT
                    );
                    
                    insert into cohort_relevant_features (item_id , _table)
                    values
                        (50889, 'labevents'),
                        (51256, 'labevents'),
                        (51279,  'labevents'),
                        (50811 ,'labevents'),
                        (51221 , 'labevents'),
                        (51250 , 'labevents'),
                        (51248 , 'labevents'),
                        (51249 , 'labevents'),
                        (51277 , 'labevents'),
                        (51244 , 'labevents'),
                        (51254 , 'labevents'),
                        (51200, 'labevents'),
                        (51146 , 'labevents'),
                        (51265 , 'labevents'),
                        (50971 , 'labevents'),
                        (50983 , 'labevents'),
                        (50912 , 'labevents'),
                        (50902 , 'labevents'),
                        (51006 , 'labevents'),
                        (50882 , 'labevents'),
                        (50868 , 'labevents'),
                        (50931 , 'labevents'),
                        (50960 , 'labevents'),
                        (50893 , 'labevents'),
                        (50970 , 'labevents'),
                        (50820 , 'labevents'),
                        (50802 , 'labevents'),
                        (50804 , 'labevents'),
                        (50821 , 'labevents'),
                        (50818 , 'labevents'),
                        (51275 , 'labevents'),
                        (51237 , 'labevents'),
                        (51274 , 'labevents'),
                        (223762 , 'chartevents'),
                        (676 , 'chartevents'),
                        (220045 , 'chartevents'),
                        (211 , 'chartevents'),
                        (220277 , 'chartevents'),
                        (646 , 'chartevents'),
                        (22018,  'chartevents'),
                        (456 , 'chartevents'),
                        (225199 , 'procedureevents_mv'),
                        (225202 , 'procedureevents_mv'),
                        (225203 , 'procedureevents_mv'),
                        (225204 , 'procedureevents_mv'),
                        (225205 , 'procedureevents_mv'),
                        (227194 , 'procedureevents_mv'),
                        (228286 , 'procedureevents_mv'),
                        (221223 , 'procedureevents_mv'),
                        (225399 , 'procedureevents_mv'),
                        (225400 , 'procedureevents_mv'),
                        (225402 , 'procedureevents_mv'),
                        (228169 , 'procedureevents_mv'),
                        (224263 , 'procedureevents_mv'),
                        (224264 , 'procedureevents_mv'),
                        (224267 , 'procedureevents_mv'),
                        (224268 , 'procedureevents_mv'),
                        (224269 , 'procedureevents_mv'),
                        (224270 , 'procedureevents_mv'),
                        (224272 , 'procedureevents_mv'),
                        (224273 , 'procedureevents_mv'),
                        (224274 , 'procedureevents_mv'),
                        (224275 , 'procedureevents_mv'),
                        (224276 , 'procedureevents_mv'),
                        (224277 , 'procedureevents_mv'),
                        (225468 , 'procedureevents_mv'),
                        (225477 , 'procedureevents_mv'),
                        (225479 , 'procedureevents_mv'),
                        (225428 , 'procedureevents_mv'),
                        (225429 , 'procedureevents_mv'),
                        (225430 , 'procedureevents_mv'),
                        (225433 , 'procedureevents_mv'),
                        (225434 , 'procedureevents_mv'),
                        (225436 , 'procedureevents_mv'),
                        (225439 , 'procedureevents_mv'),
                        (225441 , 'procedureevents_mv'),
                        (225442 , 'procedureevents_mv'),
                        (225445 , 'procedureevents_mv'),
                        (225446 , 'procedureevents_mv'),
                        (225447 , 'procedureevents_mv'),
                        (225448 , 'procedureevents_mv'),
                        (225449 , 'procedureevents_mv'),
                        (225450,  'procedureevents_mv'),
                        (225789 , 'procedureevents_mv'),
                        (225792 , 'procedureevents_mv'),
                        (225794 , 'procedureevents_mv'),
                        (225802 , 'procedureevents_mv'),
                        (225803 , 'procedureevents_mv'),
                        (225805 , 'procedureevents_mv'),
                        (225809 , 'procedureevents_mv'),
                        (227550 , 'procedureevents_mv'),
                        (227551 , 'procedureevents_mv'),
                        (225761 , 'procedureevents_mv'),
                        (224385 , 'procedureevents_mv'),
                        (225752 , 'procedureevents_mv'),
                        (227711 , 'procedureevents_mv'),
                        (227712 , 'procedureevents_mv'),
                        (227713 , 'procedureevents_mv'),
                        (227714 , 'procedureevents_mv'),
                        (224560 , 'procedureevents_mv'),
                        (224566 , 'procedureevents_mv'),
                        (226124 , 'procedureevents_mv'),
                        (227719 , 'procedureevents_mv'),
                        (228201 , 'procedureevents_mv'),
                        (228202 , 'procedureevents_mv'),
                        (225955 , 'procedureevents_mv'),
                        (226236 , 'procedureevents_mv'),
                        (226237 , 'procedureevents_mv'),
                        (226474 , 'procedureevents_mv'),
                        (226475 , 'procedureevents_mv'),
                        (226476 , 'procedureevents_mv'),
                        (226477 , 'procedureevents_mv'),
                        (225315 , 'procedureevents_mv'),
                        ( 861 , 'chartevents'),
                        (1127 , 'chartevents'),
                        (1542 , 'chartevents'),
                        (3780 , 'chartevents'),
                        (4200 , 'chartevents'),
                        (30021 , 'inputevents_cv'),
                        (45532 , 'inputevents_cv'),
                        (220546 , 'chartevents'),
                        (220955 , 'inputevents_mv'),
                        (220967 , 'inputevents_mv'),
                        (220968 , 'inputevents_mv'),
                        (220980 , 'inputevents_mv'),
                        (226780 , 'chartevents'),
                        (227062 , 'chartevents'),
                        (227063 , 'chartevents'),
                        (51349 , 'labevents'),
                        (51363 , 'labevents'),
                        (51384 , 'labevents'),
                        (51439 , 'labevents'),
                        (51458 , 'labevents'),
                        (50813 , 'labevents'),
                        (50843 , 'labevents'),
                        (50872 , 'labevents'),
                        (50954 , 'labevents'),
                        (51015 , 'labevents'),
                        (51054 , 'labevents'),
                        (51128 , 'labevents'),
                        (51232 , 'labevents'),
                        (51300 , 'labevents'),
                        (51516 , 'labevents'),
                        (51517 , 'labevents'),
                        (51518 , 'labevents'),
                        (51533 , 'labevents')
                    ;"""

        self.execute_query(query)
