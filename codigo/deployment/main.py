import json
from prediccion import predict
from preprocesamiento import FeatureEngineering


def handler(event, context):
    body = event['body']
    df = FeatureEngineering(body)
    result = predict(df)
    return result