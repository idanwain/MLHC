from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
import utils
from db_interface_eicu import DbEicu
from db_interface_mimic import DbMimic


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def get_intersected_features(db_mimic: DbMimic, db_eicu: DbEicu, threshold=0.2):
    patient_list_mimic = db_mimic.create_patient_list()
    patient_list_eicu = db_eicu.create_patient_list()

    patient_list_mimic = utils.remove_features_by_threshold(threshold, patient_list_mimic, db_mimic)
    patient_list_eicu = utils.remove_features_by_threshold(threshold, patient_list_eicu, db_eicu)

    return intersection([*(patient_list_mimic[0].events)], [*(patient_list_eicu[0].events)]), patient_list_mimic, patient_list_eicu


def get_target_list(patient_list_mimic, patient_list_eicu):
    targets_mimic = []
    targets_eicu = []
    for patient in patient_list_mimic:
        targets_mimic.append(patient.target)
    for patient in patient_list_eicu:
        targets_eicu.append(patient.target)

    return targets_mimic, targets_eicu


def fill_missing_data(labels, patient_list_mimic, patient_list_eicu, n_neighbors=10):
    data_mimic = []
    data_eicu = []

    for patient in patient_list_mimic:
        vector = patient.create_vector_for_patient(labels, True)
        data_mimic.append(vector)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    data_mimic = (imputer.fit_transform(data_mimic))

    for patient in patient_list_eicu:
        vector = patient.create_vector_for_patient(labels, True)
        data_eicu.append(vector)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    data_eicu = (imputer.fit_transform(data_eicu))

    return data_mimic, data_eicu


def feature_selection(data_mimic, data_eicu,  targets_mimic, labels, xgb_k=50):
    model = XGBClassifier()
    model.fit(data_mimic, targets_mimic)
    top_K_xgb = utils.get_top_K_features_xgb(labels, model.feature_importances_.tolist(), k=xgb_k)

    data_mimic = utils.create_vector_of_important_features(data_mimic, top_K_xgb)
    data_eicu = utils.create_vector_of_important_features(data_eicu, top_K_xgb)

    return data_mimic, data_eicu


def train_model(clf_forest, data_mimic, targets_mimic):
    clf_forest.fit(data_mimic, targets_mimic)


def model_assesment(clf_forest, data_eicu, targets_eicu):
    roc_val, pr_val = utils.calc_metrics(targets_eicu, clf_forest.predict(data_eicu))
    print("AUROC: %s\n" % roc_val)
    print("AUPR: %s\n" % pr_val)


if __name__ == "__main__":
    user = 'idan'
    data_path_eicu = 'C:/tools/feature_eicu_cohort.csv' if user == 'idan' \
        else '/Users/user/Documents/University/Workshop/feature_eicu_cohort.csv'
    boolean_features_path = 'C:/tools/boolean_features_mimic_model_a.csv' if user == 'idan'\
        else '/Users/user/Documents/University/Workshop/boolean_features_mimic_model_a.csv'
    extra_features_path = 'C:/tools/extra_features_model_a.csv' if user == 'idan'\
        else '/Users/user/Documents/University/Workshop/extra_features_model_a.csv'
    data_path_mimic = 'C:/tools/feature_mimic_cohort_model_a.csv' if user == 'idan'\
        else '/Users/user/Documents/University/Workshop/feature_mimic_cohort_model_a.csv'
    folds_path = 'C:/tools/feature_mimic_cohort_model_a.csv' if user == 'idan'\
        else '/Users/user/Documents/University/Workshop/feature_mimic_cohort_model_a.csv'

    db_mimic = DbMimic(boolean_features_path=boolean_features_path,
                    extra_features_path=extra_features_path,
                    data_path=data_path_mimic,
                    folds_path=folds_path)
    db_eicu = DbEicu(data_path=data_path_eicu)

    final_labels, patient_list_mimic, patient_list_eicu = get_intersected_features(db_mimic, db_eicu)
    targets_mimic, targets_eicu = get_target_list(patient_list_mimic, patient_list_eicu)
    data_mimic, data_eicu = fill_missing_data(final_labels, patient_list_mimic, patient_list_eicu)
    final_labels_essence_vector = utils.get_essence_label_vector(final_labels)
    data_mimic, data_eicu = feature_selection(data_mimic, data_eicu, targets_mimic, final_labels_essence_vector)

    clf_forest = RandomForestClassifier()
    train_model(clf_forest, data_mimic, targets_mimic)
    model_assesment(clf_forest, data_eicu, targets_eicu)

