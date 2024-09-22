import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.model_selection import KFold

from hiper_params_search.randomized_search_cv import RandomForestRegressorSearch, DecisionTreeRegressorSearch
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
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best', 'random'],
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': [None, 'sqrt', 'log2'],
    'max_leaf_nodes': randint(10, 100),
    'ccp_alpha': [0.0, 0.01, 0.05, 0.1]
}

hiper_parameter_searcher = DecisionTreeRegressorSearch(data_x=data_x,
                                                       data_y=data_y,
                                                       params=search_params,
                                                       cv=KFold(10),
                                                       n_jobs=-1)

randomized_search_cv = hiper_parameter_searcher.search_hipper_parameters(number_iterations=10)

cross_val_score_result = hiper_parameter_searcher.calculate_cross_val_score(searcher=randomized_search_cv)
cross_val_score_result.show_cross_val_metrics()

hiper_parameter_searcher.show_processing_time()
