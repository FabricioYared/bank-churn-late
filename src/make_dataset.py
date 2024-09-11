import pandas as pd
import numpy as np
import os

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df
def data_preparation(df):
	train = df.drop(['id', 'CustomerId', 'Surname'], axis=1)
	print("Tranformacion completa")
	return df

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('bank_default.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
       'Exited'],'bank_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('bank_new.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
       'Exited'],'bank_val.csv')
    # Matriz de Scoring
    df3 = read_file_csv('test_score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'],'bank_score_x.csv')
    
if __name__ == "__main__":
    main()
