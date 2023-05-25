import numpy as np
import pandas as pd
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

url = "https://raw.githubusercontent.com/cmd1301/coffee_ML_prediction/main/datasets/coffee_ml.csv"

df = pd.read_csv(url)

X = df.drop('Total_Cup_Points', axis=1)
y = df['Total_Cup_Points']

# separa os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# definição do modelo
model = RandomForestRegressor(n_estimators=10, max_depth=2, min_samples_leaf=120)

# pipeline e treino do modelo
categorical_cols = ['Country_of_Origin', 'Region', 'Variety', 'Processing_Method']
numerical_cols = ['Harvest_Year', 'altitude_mean_meters']

pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ]
    )),
    ('regressor', model)
])

pipeline.fit(X_train, y_train)

# faz a previsão nos dados de teste
y_pred = pipeline.predict(X_test)

# avalia a precisão do modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
y_test_mean= y_test.mean()
mae_ratio= mae/y_test_mean
rmse_ratio= rmse/y_test_mean
    
result_df = pd.DataFrame(
    data=[[mae, mse, rmse, r2, mae_ratio, rmse_ratio]], 
    columns=['MAE', 'MSE', 'RMSE', 'R2 Score', "MAE Ratio", "RMSE Ratio"])
print(result_df)

# cria a aplicação FastAPI
app = FastAPI()

# define o endpoint para a previsão
@app.post("/predict")
def predict(x1: str, x2: str, x3: float, x4: str, x5: str, x6: float):

    # transforma a entrada em um array numpy
    data = {'Country_of_Origin': [x1],
            'Region': [x2],
            'Harvest_Year': [x3],
            'Variety': [x4],
            'Processing_Method': [x5],
            'altitude_mean_meters': [x6]}
    X = pd.DataFrame(data)

    # faz a previsão usando o modelo treinado
    prediction = pipeline.predict(X)[0]

    # retorna a espécie prevista como uma resposta da API
    return {"Total Cup Points": prediction}