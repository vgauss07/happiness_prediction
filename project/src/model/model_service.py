"""
This module provides functionality for managing a ML model

It contains the ModelService class, which handles loading
and using a pretrained-ML model. The class offers methods
to load a model from a file, building it if doesn't exist,
and to make predictions from the loaded model.

"""

from pathlib import Path
import pickle as pk

from loguru import logger

from config import model_settings
from model.pipeline.model import build_model


class ModelService:
    def __init__(self):
        self.model = None

    def load_model(self):
        logger.info(f'Checking the existence of the model config file:'
                    f'{model_settings.model_path}/'
                    f'{model_settings.model_name}')

        model_path = Path(f'{model_settings.model_path}/'
                          f'{model_settings.model_name}')

        if not model_path.exists():
            logger.warning(f'model at'
                           f'{model_settings.model_path}/'
                           f'{model_settings.model_name} void'
                           f'-> building {model_settings.model_name}')
            build_model()

        logger.info(f'Model {model_settings.model_name} Exists!'
                    f'-> Load Model Configuration File')
        self.model = pk.load(
            open(f'{model_settings.model_path}/'
                 f'{model_settings.model_name}', 'rb'))

    def predict(self, input_parameters):
        logger.info('Making Prediction!')
        return self.model.predict([input_parameters])
