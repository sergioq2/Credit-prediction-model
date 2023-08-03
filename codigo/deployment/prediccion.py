import joblib
import pandas as pd
import json

model_class = 'files/model_classification.pkl'

model_class = joblib.load(model_class)


def predict(df):
    prediccion = model_class.predict(df)[0]
    if prediccion == 0:
        prediccion = "No Acepta"
    else:
        prediccion = "Si Acepta"
    probabilidad = model_class.predict_proba(df)[0].max() * 100
    return "Prediccion: " + str(prediccion) + ", Probabilidad prediccion: " + str(probabilidad) + "%"

if __name__ == '__main__':
    predict()