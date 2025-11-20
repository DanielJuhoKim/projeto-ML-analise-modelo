from pathlib import Path

import pandas as pd

from .config import (
    DATA_FOLDER,
    FILE_NAME_ORIG,
    ORIGINAL_DATASET_FOLDER,
    PROCESSED_DATASET_FOLDER,
)


def _fix_age_datatype(age_series: pd.Series) -> pd.Series:
    return age_series.astype(int)


def _fix_workclass_datatype(workclass_series: pd.Series) -> pd.Series:
    categories = [
        'Never-worked',
        'Without-pay',
        'Self-emp-inc',
        'Self-emp-not-inc',
        'Private',
        'Local-gov',
        'State-gov',
        'Federal-gov',
    ]
    cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
    return workclass_series.astype(cat_type)


def _fix_education_datatype(education_series: pd.Series) -> pd.Series:
    categories = [
        'Preschool',
        '1st-4th',
        '5th-6th',
        '7th-8th',
        '9th',
        '10th',
        '11th',
        '12th',
        'HS-grad',
        'Assoc-acdm',
        'Assoc-voc',
        'Some-college',
        'Bachelors',
        'Prof-school',
        'Masters',
        'Doctorate',
    ]
    cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
    return education_series.astype(cat_type)


def _fix_education_num_datatype(education_num_series: pd.Series) -> pd.Series:
    return education_num_series.astype(int)


def _fix_marital_status_datatype(marital_status_series: pd.Series) -> pd.Series:
    categories = [
        'Never-married',
        'Married-civ-spouse',
        'Married-AF-spouse',
        'Married-spouse-absent',
        'Separated',
        'Divorced',
        'Widowed',
    ]
    cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
    return marital_status_series.astype(cat_type)


def _fix_occupation_datatype(occupation_series: pd.Series) -> pd.Series:
    categories = [
        'Armed-Forces',
        'Priv-house-serv',
        'Protective-serv',
        'Tech-support',
        'Farming-fishing',
        'Handlers-cleaners',
        'Transport-moving',
        'Machine-op-inspct',
        'Other-service',
        'Sales',
        'Adm-clerical',
        'Exec-managerial',
        'Craft-repair',
        'Prof-specialty',
    ]
    cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
    return occupation_series.astype(cat_type)


def _fix_relationship_datatype(relationship_series: pd.Series) -> pd.Series:
    categories = [
        'Not-in-family',
        'Unmarried',
        'Other-relative',
        'Own-child',
        'Husband',
        'Wife',
    ]
    cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
    return relationship_series.astype(cat_type)


def _fix_race_datatype(race_series: pd.Series) -> pd.Series:
    categories = [
        'Other',
        'Amer-Indian-Eskimo',
        'Asian-Pac-Islander',
        'Black',
        'White',
    ]
    cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
    return race_series.astype(cat_type)


def _fix_sex_datatype(sex_series: pd.Series) -> pd.Series:
    categories = [
        'Female',
        'Male',
    ]
    cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
    return sex_series.astype(cat_type)


def _fix_capital_gain_datatype(capital_gain_series: pd.Series) -> pd.Series:
    return capital_gain_series.astype(float)


def _fix_capital_loss_datatype(capital_loss_series: pd.Series) -> pd.Series:
    return capital_loss_series.astype(float)


def _fix_hours_per_week_datatype(hours_per_week_series: pd.Series) -> pd.Series:
    return hours_per_week_series.astype(int)


def _fix_native_country_datatype(native_country_series: pd.Series) -> pd.Series:
    categories = native_country_series.value_counts().sort_values().index.tolist()
    cat_type = pd.CategoricalDtype(categories=categories, ordered=False)
    return native_country_series.astype(cat_type)


def fix_income_datatype(income_series: pd.Series) -> pd.Series:
    categories = [
        '<=50K',
        '>50K',
    ]
    cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
    return income_series.astype(cat_type)


def _fix_datatypes(dataframe: pd.DataFrame) -> pd.DataFrame:
    column_to_fixer = {
        'age': _fix_age_datatype,
        'workclass': _fix_workclass_datatype,
        'education': _fix_education_datatype,
        'education.num': _fix_education_num_datatype,
        'marital.status': _fix_marital_status_datatype,
        'occupation': _fix_occupation_datatype,
        'relationship': _fix_relationship_datatype,
        'race': _fix_race_datatype,
        'sex': _fix_sex_datatype,
        'capital.gain': _fix_capital_gain_datatype,
        'capital.loss': _fix_capital_loss_datatype,
        'hours.per.week': _fix_hours_per_week_datatype,
        'native.country': _fix_native_country_datatype,
        'income': fix_income_datatype,
    }
    for column, fixer in column_to_fixer.items():
        dataframe[column] = fixer(dataframe[column])
    return dataframe


def _fix_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.drop(columns=['fnlwgt'])
    return _fix_datatypes(dataframe)


def _read_dataframe(base_path: Path) -> pd.DataFrame:
    file_orig_path = base_path / DATA_FOLDER / ORIGINAL_DATASET_FOLDER / FILE_NAME_ORIG
    return pd.read_csv(file_orig_path, na_values='?')


def read_dataset(file_path: Path) -> pd.DataFrame:
    df = _read_dataframe(file_path)
    df = _fix_dataframe(df)
    return df


def save_processed_datasets(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_path: Path,
) -> None:
    data_folder = base_path / DATA_FOLDER / PROCESSED_DATASET_FOLDER
    data_folder.mkdir(parents=True, exist_ok=True)

    X.to_csv(data_folder / 'X.csv', index=False)
    y.to_csv(data_folder / 'y.csv', index=False)
    X_train.to_csv(data_folder / 'X_train.csv', index=False)
    X_test.to_csv(data_folder / 'X_test.csv', index=False)
    y_train.to_csv(data_folder / 'y_train.csv', index=False)
    y_test.to_csv(data_folder / 'y_test.csv', index=False)


def load_processed_datasets(base_path: Path):
    data_folder = base_path / DATA_FOLDER / PROCESSED_DATASET_FOLDER

    X = pd.read_csv(data_folder / 'X.csv')
    y = pd.read_csv(data_folder / 'y.csv').squeeze()
    X_train = pd.read_csv(data_folder / 'X_train.csv')
    X_test = pd.read_csv(data_folder / 'X_test.csv')
    y_train = pd.read_csv(data_folder / 'y_train.csv').squeeze()
    y_test = pd.read_csv(data_folder / 'y_test.csv').squeeze()

    return X, y, X_train, X_test, y_train, y_test
