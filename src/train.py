import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
import pickle
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    num_col = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    obj_col = ['Geography', 'Gender']
    min_max = MinMaxScaler()
    one_hot_encoder = OneHotEncoder()
    preprocessing = ColumnTransformer([
        ("num", min_max, num_col),
        ("obj", one_hot_encoder, obj_col),
        ('cat', 'passthrough', ['Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember'])])
    X_train = df.drop(['Exited'],axis=1)
    y_train = df[['Exited']]
    X_train1 = preprocessing.fit_transform(X_train)

    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    model = CatBoostClassifier()
    model.fit(X_train1, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(model, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('bank_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()



