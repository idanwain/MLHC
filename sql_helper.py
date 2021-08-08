from typing import List
import utils


class SqlHelper:
    def __init__(self, db_connection, model_type, username, training):
        self.cursor = db_connection.cursor()
        self.model_type = model_type
        self.user = username
        self.training = '_train_data' if training else ''

    def close(self):
        self.cursor.close()

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

    def load_cohort_to_db(self, file_path):
        query = f"""SET datestyle = dmy;
                    DROP TABLE IF EXISTS model_{self.model_type}_mimic_cohort{self.training};
                    CREATE TABLE model_{self.model_type}_mimic_cohort{self.training} (
                      identifier VARCHAR(50),
                      subject_id VARCHAR(50),
                      hadm_id VARCHAR(50),
                      admittime TIMESTAMP,
                      icu_time TIMESTAMP,
                      target_time TIMESTAMP,
                      target VARCHAR(50)
                    );
                    COPY model_{self.model_type}_mimic_cohort{self.training}
                    FROM {file_path}
                    DELIMITER ','
                    CSV HEADER;"""

        self.execute_query(query)

    def create_features_table(self):
        query = f"""DROP TABLE IF EXISTS cohort_relevant_features{self.training};
                    create table cohort_relevant_features{self.training}(
                        item_id INT,
                        _table TEXT
                    );
                    
                    insert into cohort_relevant_features{self.training} (item_id , _table)
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

    def init_boolean_features(self):
        output_path = f"'C:/tools/boolean_features_mimic_model_{self.model_type}{self.training}.csv'" if self.user == 'idan' \
            else f"'/Users/user/Documents/University/Workshop/boolean_features_mimic_model_{self.model_type}{self.training}.csv'"
        query = f"""drop table if exists boolean_features{self.training};
                    create table boolean_features{self.training} (itemid int, linksto varchar(50), category varchar(100));
                    insert into boolean_features{self.training} select itemid, linksto, category from d_items where category in
                    (
                    'Access Lines - Invasive',
                    'Lumbar Puncture',
                    'Impella',
                    'Arterial Line Insertion',
                    'IABP',
                    'Intubation',
                    'Thoracentesis',
                    'Dialysis',
                    'Access Lines - Peripheral',
                    'PA Line Insertion',
                    'Tandem Heart',
                    'PICC Line Insertion',
                    'CVL Insertion',
                    'Paracentesis'
                    ) and linksto = 'chartevents';
                    
                    insert into cohort_relevant_features{self.training} select itemid, linksto from boolean_features{self.training};
                    COPY boolean_features{self.training} To
                    {output_path}
                    With CSV DELIMITER ',' HEADER;"""

        self.execute_query(query)

    def create_symptoms(self):
        query = f"""drop table if exists relevant_note_events{self.training};
                    create table relevant_note_events{self.training} as
                        select *
                        from noteevents as ne
                        where ne.hadm_id in (select cast(hadm_id as int) from model_{self.model_type}_mimic_cohort{self.training});
                    
                    alter table relevant_note_events{self.training} drop column if exists target_time;
                    alter table relevant_note_events{self.training} add column target_time timestamp without time zone;
                    update relevant_note_events{self.training} as tr set target_time = (
                      select target_time
                      from model_{self.model_type}_mimic_cohort{self.training} as re
                      where tr.hadm_id = cast(re.hadm_id as int)
                      group by re.target_time, re.hadm_id
                    );
                    
                    alter table relevant_note_events{self.training} drop column if exists fever;
                    alter table relevant_note_events{self.training} drop column if exists chills;
                    alter table relevant_note_events{self.training} drop column if exists nausea;
                    alter table relevant_note_events{self.training} drop column if exists vomit;
                    alter table relevant_note_events{self.training} drop column if exists diarrhea;
                    alter table relevant_note_events{self.training} drop column if exists fatigue;
                    alter table relevant_note_events{self.training} drop column if exists weakness;
                    alter table relevant_note_events{self.training} drop column if exists symptoms;
                    alter table relevant_note_events{self.training} add column fever int;
                    alter table relevant_note_events{self.training} add column chills int;
                    alter table relevant_note_events{self.training} add column nausea int;
                    alter table relevant_note_events{self.training} add column vomit int;
                    alter table relevant_note_events{self.training} add column diarrhea int;
                    alter table relevant_note_events{self.training} add column fatigue int;
                    alter table relevant_note_events{self.training} add column weakness int;
                    alter table relevant_note_events{self.training} add column symptoms int;
                    
                    update relevant_note_events{self.training} set fever = case when
                        (text like '%fever%' or text like '%Fever%') and
                        target_time is not null and category in ('Case Management', 'Consult', 'General', 'Nursing', 'Nursing/other', 'Pharmacy', 'Physician')
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} set chills = case when
                        (text like '%chill%' or text like '%Chill%') and
                        target_time is not null and category in ('Case Management', 'Consult', 'General', 'Nursing', 'Nursing/other', 'Pharmacy', 'Physician')
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} set nausea = case when
                        (text like '%nausea%' or text like '%Nausea%') and
                        target_time is not null and category in ('Case Management', 'Consult', 'General', 'Nursing', 'Nursing/other', 'Pharmacy', 'Physician')
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} set vomit = case when
                        (text like '%vomit%' or text like '%Vomit%') and
                        target_time is not null and category in ('Case Management', 'Consult', 'General', 'Nursing', 'Nursing/other', 'Pharmacy', 'Physician')
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} set diarrhea = case when
                        (text like '%diarrhea%' or text like '%Diarrhea%') and
                        target_time is not null and category in ('Case Management', 'Consult', 'General', 'Nursing', 'Nursing/other', 'Pharmacy', 'Physician')
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} set fatigue = case when
                        (text like '%fatigue%' or text like '%Fatigue%') and
                        target_time is not null and category in ('Case Management', 'Consult', 'General', 'Nursing', 'Nursing/other', 'Pharmacy', 'Physician')
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} set weakness = case when
                        (text like '%weakness%' or text like '%Weakness%') and
                        target_time is not null and category in ('Case Management', 'Consult', 'General', 'Nursing', 'Nursing/other', 'Pharmacy', 'Physician')
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} as rne1 set fever =
                        case when exists(select 1 from relevant_note_events rne2 where rne1.hadm_id = rne2.hadm_id and rne2.fever = 1 group by rne1.hadm_id)
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} as rne1 set chills =
                        case when exists(select 1 from relevant_note_events rne2 where rne1.hadm_id = rne2.hadm_id and rne2.chills = 1 group by rne1.hadm_id)
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} as rne1 set nausea =
                        case when exists(select 1 from relevant_note_events rne2 where rne1.hadm_id = rne2.hadm_id and rne2.nausea = 1 group by rne1.hadm_id)
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} as rne1 set vomit =
                        case when exists(select 1 from relevant_note_events rne2 where rne1.hadm_id = rne2.hadm_id and rne2.vomit = 1 group by rne1.hadm_id)
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} as rne1 set diarrhea =
                        case when exists(select 1 from relevant_note_events rne2 where rne1.hadm_id = rne2.hadm_id and rne2.diarrhea = 1 group by rne1.hadm_id)
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} as rne1 set fatigue =
                        case when exists(select 1 from relevant_note_events rne2 where rne1.hadm_id = rne2.hadm_id and rne2.fatigue = 1 group by rne1.hadm_id)
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} as rne1 set weakness =
                        case when exists(select 1 from relevant_note_events rne2 where rne1.hadm_id = rne2.hadm_id and rne2.weakness = 1 group by rne1.hadm_id)
                        then 1 else 0 end;
                    
                    update relevant_note_events{self.training} set symptoms = case
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 0
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 1
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 2
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 3
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 4
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 5
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 6
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 7
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 8
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 9
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 10
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 11
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 12
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 13
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 14
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 0 then 15
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 16
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 17
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 18
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 19
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 20
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 21
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 22
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 23
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 24
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 25
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 26
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 27
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 28
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 29
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 30
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 0 then 31
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 32
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 33
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 34
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 35
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 36
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 37
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 38
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 39
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 40
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 41
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 42
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 43
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 44
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 45
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 46
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 0 then 47
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 48
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 49
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 50
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 51
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 52
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 53
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 54
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 55
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 56
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 57
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 58
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 59
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 60
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 61
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 62
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 0 then 63
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 64
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 65
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 66
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 67
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 68
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 69
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 70
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 71
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 72
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 73
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 74
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 75
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 76
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 77
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 78
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 0 and weakness = 1 then 79
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 80
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 81
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 82
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 83
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 84
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 85
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 86
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 87
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 88
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 89
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 90
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 91
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 92
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 93
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 94
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 0 and weakness = 1 then 95
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 96
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 97
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 98
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 99
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 100
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 101
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 102
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 103
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 104
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 105
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 106
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 107
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 108
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 109
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 110
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 0 and fatigue = 1 and weakness = 1 then 111
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 112
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 113
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 114
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 115
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 116
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 117
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 118
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 0 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 119
                        when fever = 0 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 120
                        when fever = 1 and chills = 0 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 121
                        when fever = 0 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 122
                        when fever = 1 and chills = 1 and nausea = 0 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 123
                        when fever = 0 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 124
                        when fever = 1 and chills = 0 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 125
                        when fever = 0 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 126
                        when fever = 1 and chills = 1 and nausea = 1 and vomit = 1 and diarrhea = 1 and fatigue = 1 and weakness = 1 then 127
                    end;"""

        self.execute_query(query)

    def merge_features_and_cohort(self):
        output_path = f"'C:/tools/external_validation_set_{self.model_type}{self.training}.csv'" if self.user == 'idan'\
            else f"'/Users/user/Documents/University/Workshop/external_validation_set_{self.model_type}{self.training}.csv'"
        query = f"""DROP TABLE IF EXISTS relevant_labevents_for_cohort{self.training};
                    CREATE TABLE relevant_labevents_for_cohort{self.training} as (
                        select subject_id||'-'||hadm_id as identifier, subject_id, hadm_id, itemid, charttime, value::varchar(255), valuenum, valueuom, label
                        from labevents join (select itemid, label from d_labitems) as t1 using (itemid)
                        where subject_id||'-'||hadm_id in (select identifier from model_{self.model_type}_mimic_cohort{self.training}) 
                        AND itemid in (select item_id from cohort_relevant_features{self.training} where _table='labevents')
                    );
                    
                    
                    DROP TABLE IF EXISTS relevant_chartevents_for_cohort{self.training};
                    CREATE TABLE relevant_chartevents_for_cohort{self.training} as (
                        select subject_id||'-'||hadm_id as identifier, subject_id, hadm_id, itemid, charttime, value::varchar(255), valuenum, valueuom, label
                        from chartevents join (select itemid, label from d_items) as t1 using (itemid)
                        where subject_id||'-'||hadm_id in (select identifier from model_{self.model_type}_mimic_cohort{self.training}) 
                            AND itemid in (select item_id from cohort_relevant_features{self.training} where _table='chartevents')
                    );
                    
                    DROP TABLE IF EXISTS relevant_procedure_for_cohort{self.training};
                    CREATE TABLE relevant_procedure_for_cohort{self.training} as (
                        select subject_id||'-'||hadm_id as identifier, subject_id, hadm_id, itemid, starttime as charttime, value::varchar(255), value as valuenum, valueuom, label
                        from procedureevents_mv join (select itemid, label from d_items) as t1 using (itemid)
                        where subject_id||'-'||hadm_id in (select identifier from model_{self.model_type}_mimic_cohort{self.training}) 
                            AND itemid in (select item_id from cohort_relevant_features{self.training} where _table='procedureevents_mv')
                    );
                    
                    DROP TABLE IF EXISTS relevant_inputs_mv_for_cohort{self.training};
                    CREATE TABLE relevant_inputs_mv_for_cohort{self.training} as (
                        select subject_id||'-'||hadm_id as identifier, subject_id, hadm_id, itemid, starttime as charttime, amount::varchar(255) as value, amount as valuenum, amountuom as valueuom, label
                        from inputevents_mv join (select itemid, label from d_items) as t1 using (itemid)
                        where subject_id||'-'||hadm_id in (select identifier from model_{self.model_type}_mimic_cohort{self.training}) 
                            AND itemid in (select item_id from cohort_relevant_features{self.training} where _table='inputevents_mv')
                    );
                    
                    
                    DROP TABLE IF EXISTS relevant_inputs_cv_for_cohort{self.training};
                    CREATE TABLE relevant_inputs_cv_for_cohort{self.training} as (
                        select subject_id||'-'||hadm_id as identifier, subject_id, hadm_id, itemid, charttime, amount::varchar(255) as value, amount as valuenum, amountuom as valueuom, label
                        from inputevents_cv join (select itemid, label from d_items) as t1 using (itemid)
                        where subject_id||'-'||hadm_id in (select identifier from model_{self.model_type}_mimic_cohort{self.training}) 
                            AND itemid in (select item_id from cohort_relevant_features{self.training} where _table='inputevents_cv')
                    );


                    DROP TABLE IF EXISTS relevant_note_events_for_cohort{self.training};
                    CREATE TABLE relevant_note_events_for_cohort{self.training} as (
                        select subject_id||'-'||hadm_id as identifier, subject_id, hadm_id, 0 as itemid, charttime, symptoms::varchar(255) as value, symptoms as valuenum, '' as valueuom, 'symptoms' as label
                        from relevant_note_events{self.training}
                        where subject_id||'-'||hadm_id in (select identifier from model_{self.model_type}_mimic_cohort{self.training})
                    );

                    DROP TABLE IF EXISTS relevant_drug_events_for_cohort{self.training};
                    CREATE TABLE relevant_drug_events_for_cohort{self.training} as (
                        select subject_id||'-'||hadm_id as identifier, subject_id, hadm_id, 1 as itemid, STARTDATE as charttime, '0' as value, (CASE WHEN DOSE_VAL_RX~E'^\\d+$' THEN DOSE_VAL_RX::integer ELSE RIGHT(DOSE_VAL_RX,1)::integer END) as valuenum, DOSE_UNIT_RX as valueuom, drug as label
                        from relevant_drug_events{self.training}
                        where subject_id||'-'||hadm_id in (select identifier from model_{self.model_type}_mimic_cohort{self.training})
                    );
                    
                    /* (5_c) Create a unified table of feature from the tables created above:*/
                    DROP TABLE IF EXISTS all_relevant_lab_features{self.training};
                    CREATE TABLE all_relevant_lab_features{self.training} as (
                        select * from 
                        relevant_chartevents_for_cohort{self.training} 
                        union 
                        (select * from relevant_labevents_for_cohort{self.training})
                        union
                        (select * from relevant_procedure_for_cohort{self.training})
                        union
                        (select * from relevant_inputs_mv_for_cohort{self.training})
                        union
                        (select * from relevant_inputs_cv_for_cohort{self.training})
                        union
                        (select identifier, subject_id, hadm_id, itemid, charttime, value, valuenum, valueuom, label
                        from (select *, ROW_NUMBER() OVER(PARTITION BY identifier ORDER BY charttime ASC) rn
                        from relevant_note_events_for_cohort{self.training}
                        ) x where x.rn = 1)
                        union 
                        (select * from relevant_drug_events_for_cohort{self.training})
                    );
                    
                    /* (5_d) Create a table of relevant events (features) received near when the target (culture) was received */
                    DROP TABLE IF EXISTS relevant_events{self.training};
                    CREATE TABLE relevant_events{self.training} as(
                        SELECT 
                                *,
                                date_part('year', admittime) - date_part('year', dob) as estimated_age,
                                round(CAST((extract(epoch from target_time - all_relevant_lab_features{self.training}.charttime) / 3600.0) as numeric),2) as hours_from_charttime_time_to_targettime,
                                round(CAST((extract(epoch from charttime - admittime) / 3600.0 ) as numeric),2) as hours_from_admittime_to_charttime,
                                round(CAST((extract(epoch from target_time - admittime) / 3600.0) as numeric),2) as hours_from_admittime_to_targettime
                        FROM 
                            all_relevant_lab_features{self.training}			
                            INNER JOIN (select identifier, target, target_time, admittime from model_{self.model_type}_mimic_cohort{self.training}) _tmp2 using (identifier)
                            INNER JOIN (select subject_id,gender, dob from patients where subject_id in (
                                                            select CAST (subject_id as INTEGER) 
                                                            from model_{self.model_type}_mimic_cohort{self.training})) as t3 	
                                        using (subject_id)
                        WHERE 
                             identifier in (select identifier from model_{self.model_type}_mimic_cohort{self.training})
                        AND
                            (extract(epoch from target_time - all_relevant_lab_features{self.training}.charttime)) > 0
                    );
                    
                    /* (5_e) save table to CSV file*/
                    COPY relevant_events{self.training} To
                    {output_path}
                    With CSV DELIMITER ',' HEADER;"""

        self.execute_query(query)

    def create_drug_table(self):
        query = f"""DROP TABLE IF EXISTS relevant_drug{self.training};
                    CREATE TABLE relevant_drug{self.training} AS
                    SELECT drug
                    FROM (
                        select distinct drug from prescriptions where
                        lower(drug_name_generic) like lower('%ceftriaxone%') OR
                        lower(drug_name_generic) like lower('%ciprofloxacin%') OR
                        lower(drug_name_generic) like lower('%clindamycin%') OR
                        lower(drug_name_generic) like lower('%cefotaxime%') OR
                        lower(drug_name_generic) like lower('%metronidazole%') OR
                        lower(drug_name_generic) like lower('%vancomycin%') OR
                        lower(drug_name_generic) like lower('%Cipro%') OR
                        lower(drug_name_generic) like lower('%Flagyl%') OR
                        lower(drug_name_generic) like lower('%amikacin%') OR
                        lower(drug_name_generic) like lower('%cefepime%') OR
                        lower(drug_name_generic) like lower('%Zyvox%') OR
                        lower(drug_name_generic) like lower('%ceftazidime%') OR
                        lower(drug_name_generic) like lower('%Vancocin%') OR
                        lower(drug_name_generic) like lower('%Amikin%') OR
                        lower(drug_name_generic) like lower('%Cleocin%') OR
                        lower(drug_name_generic) like lower('%piperacillin%') OR
                        lower(drug_name_generic) like lower('%tazobactam%') OR
                        lower(drug_name_generic) like lower('%Zosyn%') OR
                        lower(drug_name_generic) like lower('%Azactam%') OR
                        lower(drug_name_generic) like lower('%cilastatin%') OR
                        lower(drug_name_generic) like lower('%imipenem%') OR
                        lower(drug_name_generic) like lower('%Flagyl IV%') OR
                        lower(drug_name_generic) like lower('%gentamicin%') OR
                        lower(drug_name_generic) like lower('%linezolid%') OR
                        lower(drug_name_generic) like lower('%Maxipime%') OR
                        lower(drug_name_generic) like lower('%Claforan%') OR
                        lower(drug_name_generic) like lower('%Cipro XR%') OR
                        lower(drug_name_generic) like lower('%Cubicin%') OR
                        lower(drug_name_generic) like lower('%daptomycin%') OR
                        lower(drug_name_generic) like lower('%Garamycin%') OR
                        lower(drug_name_generic) like lower('%aztreonam%') OR
                        lower(drug_name_generic) like lower('%Cipro I.V.%') OR
                        lower(drug_name_generic) like lower('%Cleocin HCl%') OR
                        lower(drug_name_generic) like lower('%Cleocin Phosphate%') OR
                        lower(drug_name_generic) like lower('%Flagyl 375%') OR
                        lower(drug_name_generic) like lower('%Primaxin IV%') OR
                        lower(drug_name_generic) like lower('%ampicillin%') OR
                        lower(drug_name_generic) like lower('%Cleocin Pediatric%') OR
                        lower(drug_name_generic) like lower('%tobramycin%') OR
                        lower(drug_name_generic) like lower('%Fortaz%') OR
                        lower(drug_name_generic) like lower('%Tobi%') OR
                        lower(drug_name_generic) like lower('%Vancocin HCl%') OR
                        lower(drug_name_generic) like lower('%Vancocin HCl Pulvules%') OR
                        lower(drug_name_generic) like lower('%Tazicef%') OR
                        lower(drug_name_generic) like lower('%Amikin Pediatric%') OR
                        lower(drug_name_generic) like lower('%Cubicin RF%') OR
                        lower(drug_name_generic) like lower('%nafcillin%') OR
                        lower(drug_name_generic) like lower('%penicillin g potassium%') OR
                        lower(drug_name_generic) like lower('%penicillin g sodium%') OR
                        lower(drug_name_generic) like lower('%Synercid%') OR
                        lower(drug_name_generic) like lower('%dalfopristin / quinupristin%') OR
                        lower(drug_name_generic) like lower('%Pfizerpen%')
                    ) as pr;
                DROP TABLE IF EXISTS relevant_drug_events{self.training};
                create table relevant_drug_events{self.training} as 
                select * from prescriptions as presc where presc.drug in (select rd.drug from relevant_drug{self.training} as rd);
                """

        self.execute_query(query)
