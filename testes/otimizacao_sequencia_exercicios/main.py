import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from straction.normalize_training_data import get_dataframe_training_data

SEED = 1
np.random.seed(SEED)

df_training = get_dataframe_training_data()

# Pré-processamento de dados:
# Excluindo a coluna 'data', mas preservando a coluna 'peso' para otimização posterior
X = df_training.drop(['data'], axis=1)

# Variável target: 'peso'
y = X['peso']

# Features: removendo 'peso' (somente do treinamento, mas preservando para otimização)
X = X.drop(['peso'], axis=1)

# Convertendo colunas categóricas em variáveis dummy (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Escalar os dados (opcional, mas recomendável para modelos que lidam com otimização)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Definir objetivo: maximizar o peso previsto
def objective(params, model, X_template):
    # params = [peso, repeticoes]
    X_new = X_template.copy()
    X_new['peso'] = params[0]  # O peso que estamos otimizando
    X_new['repeticoes'] = params[1]  # Otimizando as repetições também

    # Remover a coluna 'peso' antes de passar para o modelo, pois ela não foi usada no treinamento
    X_new = X_new.drop(columns=['peso'])

    return -model.predict(X_new)[0]  # Negativo para maximizar


# Treinar modelo RandomForest
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

# Selecionar um exercício específico (primeira linha do conjunto de teste)
exercise_template = pd.DataFrame([X_test[0]], columns=X.columns).copy()

# Definir limites para peso e repetições (usando dados do dataframe original)
bounds = [(df_training['peso'][0] * 0.8, df_training['peso'][0] * 1.2),
          (df_training['repeticoes'][0] * 0.8, df_training['repeticoes'][0] * 1.2)]

# Otimizar os parâmetros 'peso' e 'repetições'
result = minimize(objective,
                  x0=[df_training['peso'][0], df_training['repeticoes'][0]],  # Valores iniciais da otimização
                  args=(rf, exercise_template),
                  bounds=bounds)

print(f'Peso otimizado: {result.x[0]}, Repetições otimizadas: {result.x[1]}')
