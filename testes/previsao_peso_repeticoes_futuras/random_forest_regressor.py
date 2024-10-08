import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.model_selection import KFold

from hiper_params_search.randomized_search_cv import RandomForestRegressorSearch
from straction.normalize_training_data import get_dataframe_training_data

SEED = 1
np.random.seed(SEED)

df_training = get_dataframe_training_data()

# Pré-processamento
# Convertendo categorias em números
dataframe = pd.get_dummies(df_training, columns=['treino', 'exercicio'], drop_first=True)

# Definindo features e target
data_x = dataframe.drop(['peso'], axis=1)  # Prevendo 'peso'
data_y = dataframe['peso']

search_params = {
    'n_estimators': [25, 50, 100, 200],
    'max_depth': [2, 5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['squared_error']
}

hiper_parameter_searcher = RandomForestRegressorSearch(data_x=data_x,
                                                       data_y=data_y,
                                                       params=search_params,
                                                       cv=KFold(10),
                                                       n_jobs=-1)

randomized_search_cv = hiper_parameter_searcher.search_hipper_parameters(number_iterations=10)

cross_val_score_result = hiper_parameter_searcher.calculate_cross_val_score(searcher=randomized_search_cv)
cross_val_score_result.show_cross_val_metrics()

hiper_parameter_searcher.show_processing_time()
