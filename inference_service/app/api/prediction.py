
from flask import abort, Blueprint, request
from pydantic import ValidationError

from schema.happiness import Happiness
from services.model_inference import ModelInferenceService

bp = Blueprint('prediction', __name__, url_prefix='/prediction')


@bp.get('/')
def get_prediction():
    # getting and checking the input patameters (feature values)
    try:
        happiness_features = Happiness(**request.args)
    except ValidationError:
        abort(code=400, description='Bad input params')

    # load an exisiting Ml model
    model_inference_service = ModelInferenceService()
    model_inference_service.load_model()

    # feed input parameters to the loaded ml model to get prediction
    # converts the validated input data into a format
    # that the ML model can process
    prediction = model_inference_service.predict(
        list(happiness_features.model_dump().values()),
    )

    # return prediction value
    return {'prediction': prediction}


@bp.post('/')
def get_prediction_post():
    # getting and checking the input parameters
    # requests are embedded as json when sending
    # a post request
    happiness_features = Happiness(**request.json)

    # load an existing ML model
    model_inference_service = ModelInferenceService()
    model_inference_service.load_model()

    # feed input parameters to the loaded ml model to get a prediction
    # converts the validated input data into a format that the ML model
    # can process
    prediction = model_inference_service.predict(
        list(happiness_features.model_dump().values()),
    )

    # return a prediction value
    # prediction.item() to ensure
    # flask can serialize the result properly and
    # Postman will get back clean JSON
    return {'prediction': prediction.item()}
