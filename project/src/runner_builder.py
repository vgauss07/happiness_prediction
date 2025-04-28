"""
Main application script for running the ML Model Service

This script initializes the MOodelBuilderService, trains the model
and saves the model

"""

from loguru import logger

from model.model_builder import ModelBuilderService


# ensure that logger catches any exception
@logger.catch
def main():
    """
    Run the application.

    Train the model, and save it to a
    direectory
    """
    logger.info("Running the Application")
    ml_svc = ModelBuilderService()
    ml_svc.train_model()


if __name__ == '__main__':
    main()
