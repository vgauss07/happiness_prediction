"""
Main application script for running the ML Model Service

This script initializes the MOdelService, loads the model
and makes a prediction using a pretrained model on
predefined input parameters and logs the output

"""

from loguru import logger

from model.model_service import ModelService


# ensure that logger catches any exception
@logger.catch
def main():
    """
    Run the application.

    Load the model, make a prediction on provided data,
    and log the prediction.
    """
    logger.info("Running the Application")
    ml_svc = ModelService()
    ml_svc.load_model()

    feature_values = {
        'gdp': 7.35,
        'support': 50.5,
        'healthy': 0.451,
        'freedom': 0.718,
        'corruption': 0.882
    }
    pred = ml_svc.predict(list(feature_values.values()))
    logger.info(f'Prediction: {pred}')


if __name__ == '__main__':
    main()
