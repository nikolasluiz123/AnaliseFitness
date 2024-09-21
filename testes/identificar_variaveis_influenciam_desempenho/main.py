import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from keras import Sequential, Input
from keras.src.layers import Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from straction.normalize_training_data import get_dataframe_training_data

SEED = 1
np.random.seed(SEED)

df_training = get_dataframe_training_data()

X = df_training.drop(['data', 'peso'], axis=1)
y = df_training['peso']

# Convertendo colunas categóricas em variáveis dummy (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Escalar os dados (opcional, mas recomendável para redes neurais)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Utilizando Floresta Aleatória para importâncias das features
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
importances = rf.feature_importances_

# Criar um dataframe com as importâncias das features
feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print(feature_importance)

# Plotar as importâncias das features
# plt.figure(figsize=(10, 6))
# sns.barplot(x=feature_importance, y=feature_importance.index)
# plt.title('Importância das Features - Random Forest')
# plt.show()

# --- Treinar Modelo Neural ---
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Uma saída para regressão
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

# --- Explicabilidade com SHAP ---
explainer = shap.KernelExplainer(model.predict, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])

# Plotar o resumo dos valores SHAP
shap.summary_plot(shap_values, X_test[:10])
