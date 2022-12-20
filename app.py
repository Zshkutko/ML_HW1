from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from collections import Counter
import pickle
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()

with open('Regression_car_model.pkl', 'rb') as file:
    model = pickle.load(file)

df_train = pd.read_csv('prepared_car_train.csv')


def one_hot_enc(df_train, df_test):
    columns = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    enc = OneHotEncoder(handle_unknown='ignore')

    for column in columns:
        encoder_df_train = pd.DataFrame(enc.fit_transform(df_train[[column]]).toarray())
        encoder_df_train.columns = enc.get_feature_names_out()
        df_train = df_train.drop(columns=[column])
        encoder_df_train = encoder_df_train.drop(columns=[encoder_df_train.columns[-1]])
        df_train = df_train.join(encoder_df_train)

        encoder_df_test = pd.DataFrame(enc.transform(df_test[[column]]).toarray())
        encoder_df_test.columns = enc.get_feature_names_out()
        df_test = df_test.drop(columns=[column])
        encoder_df_test = encoder_df_test.drop(columns=[encoder_df_test.columns[-1]])
        df_test = df_test.join(encoder_df_test)
    return df_train, df_test


def feature_engineering(df):
    df['h_per_l'] = df['max_power'] / (df['engine'] / 1000)
    df['year_sqrt'] = df['year'] ** 2
    df['official_dealer'] = (df['owner'].apply(lambda x: x.split(' ')[0]).isin(['First', 'Second'])) & (
                df['seller_type'] == 'Trustmark Dealer')
    df['official_dealer'] = df['official_dealer'].astype(int)
    for column in df.columns:
        test = df[column].dropna()
        max_freq = max(Counter(test), key=Counter(test).get)
        df[column] = df[column].fillna(max_freq)
    df['seats'] = df['seats'].astype(int)
    return df


def change_cols(df):
    columns = ['mileage', 'engine', 'max_power']
    d = {}

    for column in columns:
        proc_vals = []
        for i in df[column]:
            try:
                i = i.split(' ')[0]
                if i == '':
                    j = None
                else:
                    j = float(i)
            except:
                j = None
            proc_vals.append(j)
        d[column] = proc_vals

    for k in d.keys():
        df[k] = d[k]
    return df


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item = dict(item)
    df_predict = pd.DataFrame([item])
    df_predict = df_predict[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine',
                             'max_power', 'seats']]
    df_predict = change_cols(df_predict)
    df_predict = feature_engineering(df_predict)
    _, df_predict = one_hot_enc(df_train, df_predict)
    return model.predict(df_predict)[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df_columns = list(dict(list(items)[0]).keys())
    items_list = []
    for i in items:
        items_list.append(list(dict(i).values()))
    df_predict = pd.DataFrame(items_list, columns=df_columns)
    df_predict = df_predict[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine',
                             'max_power', 'seats']]
    df_predict = change_cols(df_predict)
    df_predict = feature_engineering(df_predict)
    _, df_predict = one_hot_enc(df_train, df_predict)
    return list(model.predict(df_predict))
