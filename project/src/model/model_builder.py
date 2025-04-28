"""
This module provides functionality for training a ML model

It contains the ModelBuilderService class, which trains
the ML model. The class offers methods
to train a model from a file, and saves the model.

"""


from loguru import logger

from config import model_settings
from model.pipeline.model import build_model


class ModelBuilderService:
    """
    A service class for training the ML model.

    This class provides functionalities to train the
    ML model from a specified path and save it.

    Attributes:
        model_path: Dir ML model is saved to.
        model_name: name of the saved model

    Methods:
        __init__: constructor that initializes the ModelBuilderService
        train_model: Trains the model and saves it.
    """
    def __init__(self):
        self.model = None
        self.model_path = model_settings.model_path
        self.model_name = model_settings.model_name

    def train_model(self):
        """
        Train model from a specified path,
        and save to model's directory.
        """
        logger.info(f'Checking the existence of the model config file:'
                    f'{self.model_path}/'
                    f'{self.model_name}')

        build_model()
