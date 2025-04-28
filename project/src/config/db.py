"""
This module sets up the application configuration.

It utilizes Pydantic's BaseSettings for configuration management,
allowing settings to be read from environment variables and a .env file
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine


class DbSettings(BaseSettings):
    """
    Database Configuration settings for the application

    Attributes:
        db_conn_str (str): Database connection for the application
    """
    model_config = SettingsConfigDict(env_file='config/.env',
                                      env_file_encoding='utf-8',
                                      extra='ignore')
    db_conn_str: str
    world_happiness_table_name: str


db_settings = DbSettings()

# create database engine
engine = create_engine(db_settings.db_conn_str)
