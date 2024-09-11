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
def eval_model(filename):
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
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_test = df.drop(['Exited'],axis=1)
    y_test = df[['Exited']]
    X_test = preprocessing.transform(X_test)
    y_pred_test=model.predict(X_test)
    # Generamos métricas de diagnóstico
    cm_test = confusion_matrix(y_test,y_pred_test)
    print("Matriz de confusion: ")
    print(cm_test)
    accuracy_test=accuracy_score(y_test,y_pred_test)
    print("Accuracy: ", accuracy_test)
    precision_test=precision_score(y_test,y_pred_test)
    print("Precision: ", precision_test)
    recall_test=recall_score(y_test,y_pred_test)
    print("Recall: ", recall_test)


# Validación desde el inicio
def main():
    df = eval_model('bank_val.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()