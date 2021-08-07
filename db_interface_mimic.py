from typing import Dict, List
from feature import Feature
import pandas as pd
from patient_mimic import PatientMimic
from datetime import datetime
import utils

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
                 data_path="/Users/user/Documents/University/Workshop/features_mimic.csv",
                 folds_path="/Users/user/Documents/University/Workshop/folds_mimic_model_a.csv"
                 ):
        self.boolean_features = pd.read_csv(boolean_features_path)
        self.relevant_events_data = pd.read_csv(data_path)
        self.folds_data = pd.read_csv(folds_path)
        self.available_labels_in_events = []
        self.anomaly_mapping = self.build_anomaly_mapping()

    def get_hadm_id_list(self) -> list:
        """
        Returns a list with all distinct hadm_id
        :return: hadm_id list
        """
        print("Fetching all admission id's")
        hadm_id_list = []
        for id in self.relevant_events_data["identifier"]:
            if id not in hadm_id_list:
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

    def get_events_by_hadm_id(self, hadm_id: str) -> Dict[str, List[Feature]]:
        """
        Creates a dictionary of labels and thier values, with a given hadm_id
        :param hadm_id: used to identify the patient
        :return: dictionary of labels and their values (value = Feature object)
        """
        patient_dict = {}
        relevant_rows = self.relevant_events_data.loc[lambda df: df['identifier'] == hadm_id, :]
        labels = self.get_labels()
        for label in labels:
            patient_dict[label] = []
        for row in relevant_rows.iterrows():
            label = row[1]["label"]
            time = datetime.strptime(row[1]["charttime"], '%Y-%m-%d %H:%M:%S')
            value = row[1]["valuenum"]
            unit_of_measuere = row[1]["valueuom"]
            if label in self.anomaly_mapping and (
                    value > self.anomaly_mapping[label]["max"] or value < self.anomaly_mapping[label]["min"]):
                # utils.log_dict(msg="Anomaly found", vals={"Label": label, "Value": value, "UOM": unit_of_measuere})
                continue
            feature = Feature(time=time, value=value, uom=unit_of_measuere)
            patient_dict[label].append(feature)
        return patient_dict

    def get_metadata_by_hadm_id(self, hadm_id: str):
        """
        Returns a tuple of values which are used as metadata given an hadm_id. Values can be found in Patient object.
        :param hadm_id: id of patient
        :return: tuple of metadata values
        """
        members = [attr for attr in dir(PatientMimic) if
                   not callable(getattr(PatientMimic, attr)) and not attr.startswith("__")]
        values = []
        relevant_rows = self.relevant_events_data.loc[lambda df: df['identifier'] == hadm_id, :]
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

    def get_boolean_features_by_hadm_id(self, hadm_id):
        res = {key: 0 for key in self.boolean_features['category']}
        relevant_rows = self.relevant_events_data.loc[lambda df: df['identifier'] == hadm_id, :]
        for event in relevant_rows.iterrows():
            if event[1]['itemid'] in list(self.boolean_features['itemid']):
                category = self.boolean_features.loc[lambda df: df['itemid'] == event[1]['itemid'], :]
                res[list(category['category'])[0]] = 1
        return res

    def create_patient_list(self, num_of_negatives=0):
        """
        Creates and returns list of all patients
        :return: list of patients
        """
        print("Building patient list from MIMIC...")
        hadm_id_list = self.get_hadm_id_list()
        patient_list = []
        counter = 0
        for hadm_id in hadm_id_list:
            estimated_age, gender, target = self.get_metadata_by_hadm_id(hadm_id)
            if target == 'negative':
                counter += 1
            if target == 'negative' and counter < num_of_negatives:
                continue
            symptoms = self.extract_symptoms_by_identifier(hadm_id)
            event_list = self.get_events_by_hadm_id(hadm_id)
            boolean_features = self.get_boolean_features_by_hadm_id(hadm_id)
            patient = PatientMimic(hadm_id, estimated_age, gender, symptoms, target, event_list, boolean_features)
            patient_list.append(patient)
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

    def extract_symptoms_by_identifier(self, identifier):
        relevant_row = self.relevant_events_data.loc[lambda df: df['identifier'] == identifier, :]
        relevant_row = relevant_row.loc[lambda df: df['label'] == 'symptoms', :]
        for row in relevant_row.iterrows():
            return row[1]['valuenum']
        return 0
