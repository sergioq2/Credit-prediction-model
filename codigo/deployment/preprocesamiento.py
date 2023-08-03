import joblib
import json
import pandas as pd

encoder_url = 'files/encoder.pkl'
scaler_url = 'files/scaler.pkl'

encoder = joblib.load(encoder_url)
scaler = joblib.load(scaler_url)
categorical = encoder.feature_names_in_

def load_mode(dict_mode: str):
    with open(dict_mode) as f:
        mode = json.load(f)
    return mode

def transform(df):
    df['Estado_Civil'] = ["soltero" if x == 'single' else x for x in df['Estado_Civil']]
    df['Estado_Civil'] = ["divorciado" if x == 'divorced' else x for x in df['Estado_Civil']]
    encoded_df = pd.DataFrame(
        encoder.transform(df[categorical]),
        columns=encoder.get_feature_names_out(categorical)
    )
    df = pd.concat([df.drop(categorical, axis=1), encoded_df], axis=1)

    scaled_df = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns
    )
    return scaled_df

def FeatureEngineering(json_data):
    data = json.loads(json_data)
    df = pd.DataFrame(data, index=[0])
    df = transform(df)
    return df

if __name__ == '__main__':
    FeatureEngineering()