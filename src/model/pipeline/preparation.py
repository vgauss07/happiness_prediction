import pandas as pd
from loguru import logger

from sklearn.preprocessing import LabelEncoder

from model.pipeline.collection import load_data_from_db


def prepare_data():
    logger.info("Starting Up the Preprocessing Pipeline")
    # load dataset
    data = load_data_from_db()   # noqa: WPS110
    # drop NaN values
    clean_data = drop_missing_value(data)
    # encode country col
    data_encoded = encode_country_col(clean_data)
    # drop country name column
    clean_encoded_data = drop_country_column(data_encoded)
    # rename columns
    final_df = rename_columns(clean_encoded_data)
    final_df = final_df.apply(lambda col: pd.to_numeric(col, errors='coerce')
                              if col.dtypes == 'object' else col)
    return final_df


def drop_missing_value(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info("Drop Missing Values")
    dataframe.dropna(inplace=True)
    return dataframe


def encode_country_col(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encode Country Column")
    le = LabelEncoder()
    dataframe['country_encoded'] = le.fit_transform(dataframe['Country_name'])
    return dataframe


def drop_country_column(dataframe: pd.DataFrame):
    logger.info("Drop Country Name Column")
    dataframe.drop(columns=['Country_name'], inplace=True)
    return dataframe


def rename_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info("Rename Columns")
    dataframe.rename(columns={
                        'Life_Ladder': 'score',
                        'country_encoded': 'country',
                        'Log_GDP_per_capita': 'gdp',
                        'Social_support': 'support',
                        'Healthy_life_expectancy_at_birth': 'healthy',
                        'Freedom_to_make_life_choices': 'freedom',
                        'Generosity': 'generosity',
                        'Positive_affect': 'positive_emotions_experienced',
                        'Negative_affect': 'negative_emotions_experienced',
                        'Perceptions_of_corruption': 'corruption'},
                     inplace=True)
    return dataframe


df = prepare_data()
# print(df.dtypes)
