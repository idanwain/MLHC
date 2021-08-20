from typing import Dict, List
from feature import Feature
import pandas as pd
from patient_mimic import PatientMimic
from datetime import datetime
import numpy as np
import utils
from sklearn.feature_extraction.text import CountVectorizer
import hashlib

mimic_to_eicu_mapping = {
    'Neturophils': '-polys',
    'Red Blood Cells': 'RBC',
    'Hemoglobin': 'Hgb',
    'Hematocrit': 'Hct',
    'MCV': 'MCV',
    'MCH': 'MCH',
    'MCHC': 'MCHC',
    'RDW': 'RDW',
    'Lymphocytes': '-lymphs',
    'Monocytes': '-monos',
    'Eosinophils': '-eos',
    'Basophils': '-basos',
    'Platelet Count': 'platelets x 1000',

    'Potassium': 'potassium',
    'Sodium': 'sodium',
    'Creatinine': 'creatinine',
    'Chloride': 'chloride',
    'Urea Nitrogen': 'BUN',
    'Bicarbonate': 'bicarbonate',
    'Anion Gap': 'anion gap',
    'Glucose': 'glucose',
    'Magnesium': 'magnesium',
    'Calcium, Total': 'calcium',
    'Phosphate': 'phosphate',
    'pH': 'pH',

    'Base Excess': 'Base Excess',
    'Calculated Total CO2': 'Total CO2',
    'pO2': 'paO2',
    'pCO2': 'paCO2',

    'PTT': 'PTT',
    'INR(PT)': 'PT - INR',
    'PT': 'PT'
}


class DbMimic:
    def __init__(self,
                 boolean_features_path,
                 mimic_data_path="/Users/user/Documents/University/Workshop/features_mimic.csv",
                 folds_path=None,
                 eicu_data_path=None
                 ):
        self.boolean_features = pd.read_csv(boolean_features_path)
        self.itemid_to_category_mapping = self.map_itemid_to_category()
        data = [pd.read_csv(mimic_data_path)]
        folds = []
        if folds_path:
            folds = [pd.read_csv(folds_path)]
        if eicu_data_path:
            data.append(pd.read_csv(eicu_data_path))
            if folds_path:
                eicu_fold = pd.DataFrame({'identifier': data[1][["identifier"]]["identifier"].unique()})
                for i in range(0, 5):
                    eicu_fold.loc[(i / 5) * len(eicu_fold):(i + 1 / 5) * len(eicu_fold), 'fold'] = i + 1
                folds.append(eicu_fold)
        if folds_path:
            self.folds_data = pd.concat(folds)
        self.relevant_events_data = pd.concat(data)
        self.anomaly_mapping = self.build_anomaly_mapping()
        self.available_labels_in_events = []
        self.available_drugs = []
        self.invasive_procedures = []

    def get_hadm_id_list(self) -> list:
        """
        Returns a list with all distinct hadm_id
        :return: hadm_id list
        """
        print("Fetching all admission id's")
        hadm_id_list = []
        for id in self.relevant_events_data["identifier"].unique():
            hadm_id_list.append(id)
        return hadm_id_list

    def get_labels(self) -> list:
        """
        Returns a list with all distinct labels
        :return: labels list
        """
        distinct_labels = []
        if (len(self.available_labels_in_events) == 0):
            for label in self.relevant_events_data["label"]:
                if (label not in distinct_labels):
                    distinct_labels.append(label)
            self.available_labels_in_events = distinct_labels
            return distinct_labels
        else:
            return self.available_labels_in_events

    def get_drugs(self) -> list:
        """
        Returns a list with all distinct drugs
        :return: drugs list
        """
        distinct_drugs = []
        if (len(self.available_drugs) == 0):
            relevant_row = self.relevant_events_data.loc[lambda df: df['itemid'] == 1, :]
            for row in relevant_row.iterrows():
                distinct_drugs.append(row[1]['label'])
            self.available_drugs = [' '.join(list(set(distinct_drugs)))]
        return self.available_drugs

    def get_invasive_procedures(self) -> list:
        """
        Returns a list with all distinct drugs
        :return: drugs list
        """
        distinct_procedures = []
        if (len(self.invasive_procedures) == 0):
            boolean_features_list = list(self.boolean_features['itemid'])
            relevant_rows = self.relevant_events_data[self.relevant_events_data['itemid'].isin(boolean_features_list)]
            for row in relevant_rows.iterrows():
                distinct_procedures.append(hashlib.sha1(row[1]['label'].encode()).hexdigest())
            self.invasive_procedures = [' '.join(list(set(distinct_procedures)))]
        return self.invasive_procedures

    def get_events_by_hadm_id(self, hadm_id: str, relevant_rows) -> Dict[str, List[Feature]]:
        """
        Creates a dictionary of labels and thier values, with a given hadm_id
        :param hadm_id: used to identify the patient
        :return: dictionary of labels and their values (value = Feature object)
        """
        patient_dict = {}
        relevant_rows = relevant_rows
        labels = self.get_labels()
        for label in labels:
            patient_dict[label] = []
        for row in relevant_rows.iterrows():
            label = row[1]["label"]
            time = datetime.strptime(row[1]["charttime"], '%Y-%m-%d %H:%M:%S')
            value = row[1]["valuenum"]
            if np.isnan(value):
                value = self.parse_value(row[1]['value'])
            unit_of_measuere = row[1]["valueuom"]
            if (label in self.anomaly_mapping and
                (value > self.anomaly_mapping[label]["max"] or
                 value < self.anomaly_mapping[label]["min"])) or \
                    np.isnan(value):
                continue
            feature = Feature(time=time, value=value, uom=unit_of_measuere)
            patient_dict[label].append(feature)
        return patient_dict

    def get_metadata_by_hadm_id(self, hadm_id: str, relevant_rows, members):
        """
        Returns a tuple of values which are used as metadata given an hadm_id. Values can be found in Patient object.
        :param relevant_rows:
        :param hadm_id: id of patient
        :return: tuple of metadata values
        """
        values = []
        for member in members:
            values.append(relevant_rows.iloc[0][member])
        return tuple(values)

    def get_folds(self):
        """
        Returns a dictionary of size 5 containing all folds for the cross validation
        :return:
        """
        res = {}
        for row in self.folds_data.iterrows():
            identifier = row[1]["identifier"]
            fold = "fold_" + str(row[1]["fold"])
            hadm_id = identifier
            try:
                res[fold].append(hadm_id)
            except KeyError:
                res.update({fold: [hadm_id]})
        return res

    def get_boolean_features_by_hadm_id(self, hadm_id, relevant_rows):
        res = {key: 0 for key in self.boolean_features['category']}
        boolean_features_itemid_list = list(self.boolean_features['itemid'])
        relevant_rows = relevant_rows[relevant_rows['itemid'].isin(boolean_features_itemid_list)]
        for event in relevant_rows.iterrows():
            # category = self.boolean_features.loc[lambda df: df['itemid'] == event[1]['itemid'], :]
            res[self.itemid_to_category_mapping[event[1]['itemid']]] = 1
        return res

    def create_patient_list(self):
        """
        Creates and returns list of all patients
        :return: list of patients
        """
        print("Building patient list from MIMIC...")
        hadm_id_list = self.get_hadm_id_list()
        drugs_vectorizer = CountVectorizer()
        invasive_procedures_vectorizer = CountVectorizer()
        drugs_vectorizer.fit(self.get_drugs())
        invasive_procedures_vectorizer.fit(self.get_invasive_procedures())
        members = [attr for attr in dir(PatientMimic) if
                   not callable(getattr(PatientMimic, attr)) and not attr.startswith("__")]

        patient_list = []
        for hadm_id in hadm_id_list:
            relevant_rows = self.relevant_events_data.loc[lambda df: df['identifier'] == hadm_id, :]
            estimated_age, gender, target = self.get_metadata_by_hadm_id(hadm_id, relevant_rows, members)
            symptoms = self.extract_symptoms_by_identifier(hadm_id,relevant_rows)
            drugs = self.extract_drugs_by_identifier(hadm_id,relevant_rows)
            drugs_vector = sorted(list((drugs_vectorizer.transform(drugs)).toarray()[0]))
            procedures = self.extract_invasive_procedure_by_identifier(hadm_id, relevant_rows)
            procedures_vector = sorted(list((invasive_procedures_vectorizer.transform(procedures)).toarray()[0]))
            event_list = self.get_events_by_hadm_id(hadm_id, relevant_rows)
            boolean_features = self.get_boolean_features_by_hadm_id(hadm_id,relevant_rows)
            patient = PatientMimic(hadm_id, estimated_age, gender, symptoms, target, event_list, boolean_features,
                                   drugs_vector, procedures_vector)
            patient_list.append(patient)
            print(len(patient_list))
        print("Done")
        return patient_list

    def get_distinct_boolean_features(self) -> list:
        """
        Returns a list with all distinct boolean features names
        :return: labels list
        """
        return list({key for key in self.boolean_features['category']})

    def build_anomaly_mapping(self):
        data = pd.read_csv('human_range.csv')
        res = {}
        for row in data.iterrows():
            feature = row[1]["feature"]
            max_val = row[1]["max"]
            min_val = row[1]["min"]
            res[feature] = {"min": min_val, "max": max_val}
        return res

    def extract_symptoms_by_identifier(self, identifier, relevant_rows):
        relevant_row = relevant_rows
        relevant_row = relevant_row.loc[lambda df: df['label'] == 'symptoms', :]
        for row in relevant_row.iterrows():
            return row[1]['valuenum']
        return 0

    def extract_drugs_by_identifier(self, identifier,relevant_rows):
        drugs = []
        relevant_row = relevant_rows
        relevant_row = relevant_row.loc[lambda df: df['itemid'] == 1, :]
        for row in relevant_row.iterrows():
            drugs.append(row[1]['label'])
        return [' '.join(drugs)]

    def extract_invasive_procedure_by_identifier(self, identifier, relevant_rows):
        procedures = []
        boolean_features_itemid_list = list(self.boolean_features['itemid'])
        relevant_row = relevant_rows[relevant_rows['itemid'].isin(boolean_features_itemid_list)]
        for row in relevant_row.iterrows():
            procedures.append(hashlib.sha1(row[1]['label'].encode()).hexdigest())
        return [' '.join(procedures)]

    def parse_value(self, value: str):
        value = str(value)
        try:
            if value == 'nan':
                return np.nan
            if '>' in value or '<' in value:
                if '*' == value[-1]:
                    value = value[:-1]
                return float(value[1:])
            elif '-' in value:
                res = value.split('-')
                low, high = float(res[0]), float(res[1])
                return np.average([low, high])
            else:
                return np.nan
        except Exception as e:
            return np.nan

    def map_itemid_to_category(self):
        res = {}
        for row in self.boolean_features.iterrows():
            res[row[1]['itemid']] = row[1]['category']
        return res

