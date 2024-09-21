import numpy as np
from keras import Sequential, Input
from keras.src.layers import LSTM, Dense
from keras.src.legacy.preprocessing.sequence import TimeseriesGenerator
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from straction.normalize_training_data import get_dataframe_training_data

SEED = 1
np.random.seed(SEED)

df_training = get_dataframe_training_data()

# Ordenar dados por data
df_sorted = df_training.sort_values('data')

# Selecionar uma métrica para previsão, por exemplo, peso médio por dia
daily_weight = df_sorted.groupby('data')['peso'].mean().values

# Normalizar
scaler = MinMaxScaler()
daily_weight_scaled = scaler.fit_transform(daily_weight.reshape(-1, 1))

# Criar sequências para LSTM
sequence_length = 5
generator = TimeseriesGenerator(daily_weight_scaled, daily_weight_scaled, length=sequence_length, batch_size=1)

# Converter o gerador em arrays para treino e teste
X, y = zip(*[generator[i] for i in range(len(generator))])
X = np.array(X)
y = np.array(y)

# Ajustar a forma de X para (batch_size, time_steps, features)
X = X.reshape(
    (X.shape[0], sequence_length, 1))  # Garantir que os dados tenham a forma (batch_size, time_steps, features)

# Dividir em treino e teste (80% treino, 20% teste)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Construir modelo LSTM
model = Sequential()
model.add(Input(shape=(sequence_length, 1)))  # (time_steps, features)
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar
model.fit(X_train, y_train, epochs=100, verbose=0)

# Prever
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Redimensionar y_test para 2D
y_test = y_test.reshape(-1, 1)

# Inverter a transformação
y_test_actual = scaler.inverse_transform(y_test)

# Plotar resultados
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Real')
plt.plot(predictions, label='Previsto')
plt.legend()
plt.show()
