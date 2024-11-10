import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Carregar os dados
data = fetch_california_housing(as_frame=True)
df = data.frame  # Transformar em DataFrame para facilitar a manipulação

# Visualizar o dataset
print(df.head())
print(df.describe())

# Verificar se há valores nulos no DataFrame
print(df.isnull().sum())  # Aqui você deve usar df, não data


# Plotar o histograma da variável alvo (valor médio das casas) E
# Multiplicar o valor médio das casas para escala real em dólares
(df["MedHouseVal"] * 100000).hist(bins=50)  # Multiplicar os valores por 100,000
plt.xlabel("Valor Médio das Casas (em dólares)")
plt.ylabel("Número de Bairros")
plt.title("Distribuição do Valor Médio das Casas na Califórnia")
plt.grid(visible=True, alpha=0.3)
plt.show()



# Separar variáveis independentes (X) e dependente (y)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inicializar e treinar os modelos
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regression": SVR()
}

results = {}
for name, model in models.items():
    # Treinar o modelo
    model.fit(X_train, y_train)
    # Fazer previsões no conjunto de teste
    predictions = model.predict(X_test)
    # Calcular o RMSE
    rmse = sqrt(mean_squared_error(y_test, predictions))
    results[name] = rmse
    print(f"{name}: RMSE = {rmse:.2f}")

# Classificar os modelos por desempenho, do melhor para o pior
sorted_results = sorted(results.items(), key=lambda x: x[1])
print("\nClassificação dos Modelos (do melhor para o pior):")
for rank, (name, rmse) in enumerate(sorted_results, start=1):
    print(f"{rank}. {name} - RMSE: {rmse:.2f}")



