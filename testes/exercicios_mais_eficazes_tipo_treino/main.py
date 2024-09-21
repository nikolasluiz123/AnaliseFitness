import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.src.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from straction.normalize_training_data import get_dataframe_training_data

SEED = 1
np.random.seed(SEED)

# Supondo que get_dataframe_training_data retorna o dataframe correto
df_training = get_dataframe_training_data()

# Supondo que 'treino' é a classe a ser prevista
X = df_training.drop(['treino'], axis=1)
y = df_training['treino']

# Verificando colunas categóricas que precisam de conversão
categorical_cols = X.select_dtypes(include=['object']).columns

# Convertendo as colunas categóricas em variáveis dummies (OneHotEncoding)
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Convertendo a classe de treino para números
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# Identificar as classes presentes no conjunto de teste
unique_classes = np.unique(y_test)

# --- Floresta Aleatória ---
rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print('Random Forest Classification Report:')
print(classification_report(y_test, rf_pred, labels=unique_classes, target_names=le.inverse_transform(unique_classes)))

# --- Árvore de Decisão ---
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print('Decision Tree Classification Report:')
print(classification_report(y_test, dt_pred, labels=unique_classes, target_names=le.inverse_transform(unique_classes)))

# --- SVC ---
# svc = SVC(kernel='linear', probability=True)
# svc.fit(X_train, y_train)
# svc_pred = svc.predict(X_test)
# print('SVC Classification Report:')
# print(classification_report(y_test, svc_pred, labels=unique_classes, target_names=le.inverse_transform(unique_classes)))

# --- Rede Neural ---
model_clf = Sequential()
model_clf.add(Input(shape=(X_train.shape[1],)))
model_clf.add(Dense(64, activation='relu'))
model_clf.add(Dense(128, activation='relu'))
model_clf.add(Dense(64, activation='relu'))
model_clf.add(Dense(len(le.classes_), activation='softmax'))  # Saída para multiclasse

model_clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_clf.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

nn_pred_prob = model_clf.predict(X_test)
nn_pred = nn_pred_prob.argmax(axis=1)
print('Neural Network Classification Report:')
print(classification_report(y_test, nn_pred, labels=unique_classes, target_names=le.inverse_transform(unique_classes)))
