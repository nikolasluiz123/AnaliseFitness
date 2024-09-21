import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.src.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from straction.normalize_training_data import get_dataframe_training_data

SEED = 1
np.random.seed(SEED)

df_training = get_dataframe_training_data()

# Definir label: Eficaz se peso > média ou repetições > média
df_training['eficaz'] = df_training.apply(lambda row: 1 if (row['peso'] > df_training['peso'].mean()) or 
                                     (row['repeticoes'] > df_training['repeticoes'].mean()) else 0, axis=1)

# OneHotEncoding para as colunas 'treino' e 'exercicio'
X = df_training.drop(['eficaz', 'peso'], axis=1)  # Excluindo 'peso' para não viciar
y = df_training['eficaz']

# Aplicando OneHotEncoder em colunas categóricas
X = pd.get_dummies(X, columns=['treino', 'exercicio'], drop_first=True)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- Floresta Aleatória ---
rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print('Random Forest Classification Report:')
print(classification_report(y_test, rf_pred))

# --- Árvore de Decisão ---
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print('Decision Tree Classification Report:')
print(classification_report(y_test, dt_pred))

# --- SVC ---
# svc = SVC(kernel='linear', verbose=1)
# svc.fit(X_train, y_train)
# svc_pred = svc.predict(X_test)
# print('SVC Classification Report:')
# print(classification_report(y_test, svc_pred))

# --- Rede Neural ---
model_clf = Sequential()
model_clf.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model_clf.add(Dense(32, activation='relu'))
model_clf.add(Dense(1, activation='sigmoid'))  # Saída para classificação binária

model_clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_clf.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

nn_pred_prob = model_clf.predict(X_test)
nn_pred = (nn_pred_prob > 0.5).astype(int)
print('Neural Network Classification Report:')
print(classification_report(y_test, nn_pred))
