import pandas as pd
from flatbuffers.encode import np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from straction.normalize_training_data import get_dataframe_training_data

SEED = 1
np.random.seed(SEED)

df_training = get_dataframe_training_data()

# Pré-processamento
# Convertendo categorias em números
df = pd.get_dummies(df_training, columns=['treino', 'exercicio'], drop_first=True)

# Definindo features e target
X = df.drop(['peso'], axis=1)  # Prevendo 'peso'
y = df['peso']

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- Floresta Aleatória ---
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f'Random Forest MSE: {mean_squared_error(y_test, rf_pred)}')

# --- Árvore de Decisão ---
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print(f'Decision Tree MSE: {mean_squared_error(y_test, dt_pred)}')

# --- Support Vector Regressor ---
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)
print(f'SVR MSE: {mean_squared_error(y_test, svr_pred)}')
