import numpy as np
import pandas as pd
from keras import Input, Model
from keras.src.layers import Embedding, Flatten, Concatenate, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from straction.normalize_training_data import get_dataframe_training_data

SEED = 1
np.random.seed(SEED)

df_training = get_dataframe_training_data()

# Supondo que você tenha IDs para 'treino' e 'exercicio'
df_training['treino_id'] = LabelEncoder().fit_transform(df_training['treino'])
df_training['exercicio_id'] = LabelEncoder().fit_transform(df_training['exercicio'])

# Features e target
X = df_training[['treino_id', 'exercicio_id', 'serie', 'repeticoes']]
y = df_training['peso']

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Input layers
input_treino = Input(shape=(1,))
input_exercicio = Input(shape=(1,))
input_serie = Input(shape=(1,))
input_repeticoes = Input(shape=(1,))

# Embeddings para categorias
embedding_treino = Embedding(input_dim=df_training['treino_id'].nunique(), output_dim=8)(input_treino)
embedding_treino = Flatten()(embedding_treino)

embedding_exercicio = Embedding(input_dim=df_training['exercicio_id'].nunique(), output_dim=8)(input_exercicio)
embedding_exercicio = Flatten()(embedding_exercicio)

# Concatenar todas as features
concatenated = Concatenate()([embedding_treino, embedding_exercicio, input_serie, input_repeticoes])

# Camadas densas
dense = Dense(64, activation='relu')(concatenated)
dense = Dense(32, activation='relu')(dense)
output = Dense(1)(dense)

# Modelo
model = Model(inputs=[input_treino, input_exercicio, input_serie, input_repeticoes], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar
model.fit([X_train['treino_id'], X_train['exercicio_id'], X_train['serie'], X_train['repeticoes']],
          y_train, epochs=100, batch_size=10, verbose=0)

# Previsão
predictions = model.predict([X_test['treino_id'], X_test['exercicio_id'], X_test['serie'], X_test['repeticoes']])

# Avaliar o modelo
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Exibir as métricas de desempenho
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Mostrar algumas previsões vs valores reais
comparison = pd.DataFrame({
    'Real': y_test.values,
    'Previsto': predictions.flatten()
})

print("\nPrimeiras 10 previsões vs valores reais:")
print(comparison.head(10))
