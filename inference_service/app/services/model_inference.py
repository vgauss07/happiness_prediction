"""
This module provides functionality for managing a ML model

It contains the ModelInferenceService class, which handles loading
and using a pretrained-ML model. The class offers methods
to load a model from a file, building it if doesn't exist,
and to make predictions from the loaded model.

"""

from pathlib import Path
import pickle as pk

from loguru import logger

from config import model_settings


class ModelInferenceService:
    """
    A service class for making predictions

    This class provides functionalities to load ML model,
    from a specified path, built it if doesn't exist, and
    and make predictions using the loaded model.

    Attributes:
        model: ML model managed by this service. Inititally set to None.

    Methods:
        __init__: Constructor that initializes the ModelInferenceService
        load_model: loads the model from file
        predict: Makes a prediction using the loaded model

    """
    def __init__(self) -> None:
        self.model = None
        self.model_path = model_settings.model_path
        self.model_name = model_settings.model_name

    def load_model(self) -> None:
        """Initialize the ModelInferenceService with no model loaded"""
        logger.info(f'Checking the existence of the model config file:'
                    f'{self.model_path}/'
                    f'{self.model_name}')

        model_path = Path(f'{self.model_path}/'
                          f'{self.model_name}')

        if not model_path.exists():
            raise FileNotFoundError('Model file does not exist!')

        logger.info(f'Model {self.model_name} Exists!'
                    f'-> Load Model Configuration File')
        self.model = pk.load(
            open(f'{self.model_path}/'
                 f'{self.model_name}', 'rb'))

    def predict(self, input_parameters: list) -> list:
        """
        Make prediction using the loaded model

        Take input parameters and passes it to the model,
        which was loaded using a pickle file

        Args:
            input_parameters (list): The input data for making a prediction

        Returns:
            list: The prediction result from the model.
        """
        logger.info('Making Prediction!')
        return self.model.predict([input_parameters])
