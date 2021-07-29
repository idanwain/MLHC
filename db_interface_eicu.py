from typing import Dict, List
from feature import Feature
import pandas as pd
from patient_eicu import PatientEicu
import utils
eicu_to_mimic_mapping = {
    '-polys': 'Neturophils' ,
    'RBC': 'Red Blood Cells',
    'Hgb': 'Hemoglobin',
    'Hct': 'Hematocrit',
    'MCV' : 'MCV',
    'MCH': 'MCH',
    'MCHC': 'MCHC',
    'RDW': 'RDW',
    '-lymphs': 'Lymphocytes',
    '-monos': 'Monocytes',
    '-eos': 'Eosinophils',
    '-basos': 'Basophils',
    'platelets x 1000': 'Platelet Count',

    'potassium': 'Potassium',
    'sodium': 'Sodium',
    'creatinine': 'Creatinine',
    'chloride': 'Chloride',
    'BUN': 'Urea Nitrogen',
    'bicarbonate': 'Bicarbonate',
    'anion gap': 'Anion Gap',
    'glucose': 'Glucose',
    'magnesium': 'Magnesium',
    'calcium': 'Calcium, Total',
    'phosphate': 'Phosphate',
    'pH': 'pH',

    'Base Excess': 'Base Excess',
    'Total CO2': 'Calculated Total CO2',
    'paO2': 'pO2',
    'paCO2': 'pCO2',

    'PTT': 'PTT',
    'PT - INR': 'INR(PT)',
    'PT': 'PT'
}


class DbEicu:
    def __init__(self, data_path="C:/tools/feature_eicu_cohort.csv"):
        self.relevant_events_data = pd.read_csv(data_path)
        self.available_labels_in_events = []
        # self.anomaly_mapping = self.build_anomaly_mapping()


    def get_patient_health_system_stay_id_list(self) -> list:
        """
        Returns a list with all distinct patient_health_system_stay_id
        :return: patient_health_system_stay_id list
        """
        print("Fetching all admission id's")
        patient_health_system_stay_id_list = []
        for id in self.relevant_events_data["patienthealthsystemstayid"]:
            if id not in patient_health_system_stay_id_list:
                patient_health_system_stay_id_list.append(id)
        return patient_health_system_stay_id_list

    def get_labels(self) -> list:
        """
        Returns a list with all distinct labels
        :return: labels list
        """
        distinct_labels = []
        if(len(self.available_labels_in_events) == 0):
            for label in self.relevant_events_data["labname"]:
                label = eicu_to_mimic_mapping[label]
                if (label not in distinct_labels):
                    distinct_labels.append(label)
            self.available_labels_in_events = distinct_labels
            return distinct_labels
        else:
            return self.available_labels_in_events

    def get_events_by_patient_health_system_stay_id(self, patient_health_system_stay_id: str) -> Dict[str, List[Feature]]:
        """
        Creates a dictionary of labels and thier values, with a given hadm_id
        :param patient_health_system_stay_id: used to identify the patient
        :return: dictionary of labels and their values (value = Feature object)
        """
        patient_dict = {}
        relevant_rows = self.relevant_events_data.loc[lambda df: df['patienthealthsystemstayid'] == patient_health_system_stay_id, :]
        labels = self.get_labels()
        for label in labels:
            patient_dict[label] = []
        for row in relevant_rows.iterrows():
            label = row[1]["labname"]
            time = row[1]["lab_time"]
            value = row[1]["labresult"] if row[1]["labresult"] else row[1]["labresulttext"]
            unit_of_measure = row[1]["labmeasurenamesystem"] if row[1]["labmeasurenamesystem"] else row[1]["labmeasurenameinterface"]
            # if (eicu_to_mimic_mapping[label] in self.anomaly_mapping and (
            #         value > self.anomaly_mapping[label]["max"] or value < self.anomaly_mapping[label]["min"])):
            #     utils.log_dict(msg="Anomaly found", vals={"Label": label, "Value": value})
            #     continue
            feature = Feature(time=time, value=value, uom=unit_of_measure)
            patient_dict[eicu_to_mimic_mapping[label]].append(feature)
        return patient_dict

    def get_target_by_patient_health_system_stay_id(self, patient_health_system_stay_id: str):
        """
        Returns a tuple of values which are used as metadata given an hadm_id. Values can be found in Patient object.
        :param patient_health_system_stay_id: id of patient
        :return: tuple of metadata values
        """
        relevant_rows = self.relevant_events_data.loc[lambda df: df['patienthealthsystemstayid'] == patient_health_system_stay_id, :]
        return relevant_rows.iloc[0]['target']

    def create_patient_list(self):
        """
        Creates and returns list of all patients
        :return: list of patients
        """
        print("Building patient list from EICU...")
        patient_health_system_stay_id_list = self.get_patient_health_system_stay_id_list()
        patient_list = []
        i = 0
        for patient_health_system_stay_id in patient_health_system_stay_id_list:
            if i == 1500:
                break
            target = self.get_target_by_patient_health_system_stay_id(patient_health_system_stay_id)
            event_list = self.get_events_by_patient_health_system_stay_id(patient_health_system_stay_id)
            patient = PatientEicu(patient_health_system_stay_id, target, event_list)
            patient_list.append(patient)
            i += 1
        print("Done")
        return patient_list

    def build_anomaly_mapping(self):
        data = pd.read_csv('human_range.csv')
        res = {}
        for row in data.iterrows():
            feature = row[1]["feature"]
            max_val = row[1]["max"]
            min_val = row[1]["min"]
            res[feature] = {"min":min_val,"max":max_val}
        return res



