import numpy as np
from keras import Sequential, Input
from keras.src.layers import Dense
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from straction.normalize_training_data import get_dataframe_training_data

SEED = 1
np.random.seed(SEED)

df_training = get_dataframe_training_data()

# Preparar dados
X = df_training.drop(['data', 'treino', 'exercicio'], axis=1)  # Usando métricas numéricas

# Treinar modelo
iso_forest = IsolationForest(contamination=0.05)
iso_forest.fit(X)

# Prever anomalias
df_training['anomaly'] = iso_forest.predict(X)
df_training['anomaly'] = df_training['anomaly'].map({1: 0, -1: 1})  # 1 para anomalia

# Visualizar anomalias
anomalies = df_training[df_training['anomaly'] == 1]
print(anomalies)

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Construção do Autoencoder
autoencoder = Sequential()
autoencoder.add(Input(shape=(X_scaled.shape[1],)))
autoencoder.add(Dense(64, activation='relu'))
autoencoder.add(Dense(32, activation='relu'))
autoencoder.add(Dense(64, activation='relu'))
autoencoder.add(Dense(X_scaled.shape[1], activation='linear'))

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=10, shuffle=True, verbose=0)

# Reconstrução e erro
X_pred = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

# Definir threshold
threshold = np.percentile(mse, 95)
df_training['anomaly'] = (mse > threshold).astype(int)

# Visualizar anomalias
anomalies = df_training[df_training['anomaly'] == 1]
print(anomalies)