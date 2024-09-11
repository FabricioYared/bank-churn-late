import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import *


# Cargar la tabla transformada
def score_model(filename, scores):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    df1 = pd.read_csv(os.path.join('../data/processed/bank_train.csv'))
    num_col = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    obj_col = ['Geography', 'Gender']
    min_max = MinMaxScaler()
    one_hot_encoder = OneHotEncoder()
    preprocessing = ColumnTransformer([
        ("num", min_max, num_col),
        ("obj", one_hot_encoder, obj_col),
        ('cat', 'passthrough', ['Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember'])])
    X_train = df1.drop(['Exited'],axis=1)
    X_train1 = preprocessing.fit_transform(X_train)
    X_test = preprocessing.transform(df)
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de Scoring    
    res = model.predict(X_test)
    pred = pd.DataFrame(res, columns=['PREDICT'])
    pred.to_csv(os.path.join('../data/scores/', scores))
    print(scores, 'exportado correctamente en la carpeta scores')


# Scoring desde el inicio
def main():
    df = score_model('bank_score_x.csv','final_score.csv')
    print('Finalizó el Scoring del Modelo')


if __name__ == "__main__":
    main()