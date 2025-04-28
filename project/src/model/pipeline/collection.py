import pandas as pd

from loguru import logger
from sqlalchemy import select

from config import engine
from db.db_model import WorldHappiness


def load_data(FILEPATH):
    logger.info(f"Loading csv file at path {FILEPATH}")
    return pd.read_csv(FILEPATH, encoding='ISO-8859-1')


def load_data_from_db():
    logger.info("Extracting Data from the Database")
    query = select(WorldHappiness)

    df = pd.read_sql(query, engine)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).apply(lambda x: x.encode('ISO-8859-1',
                                            errors='ignore').
                                            decode('ISO-8859-1'))
    return df
