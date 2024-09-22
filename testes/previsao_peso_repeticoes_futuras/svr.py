import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.model_selection import KFold

from hiper_params_search.randomized_search_cv import RandomForestRegressorSearch, DecisionTreeRegressorSearch, SVRSearch
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
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    # 'C': np.logspace(-3, 3, 7),
    'gamma': ['scale', 'auto'],
    # 'degree': randint(1, 10),
    # 'epsilon': np.logspace(-3, 1, 5),
    # 'coef0': np.linspace(0, 1, 5),
    'shrinking': [True, False],
    # 'tol': np.logspace(-5, -1, 5)
}

hiper_parameter_searcher = SVRSearch(data_x=data_x,
                                     data_y=data_y,
                                     params=search_params,
                                     cv=KFold(5))

randomized_search_cv = hiper_parameter_searcher.search_hipper_parameters(number_iterations=10)

cross_val_score_result = hiper_parameter_searcher.calculate_cross_val_score(searcher=randomized_search_cv)
cross_val_score_result.show_cross_val_metrics()

hiper_parameter_searcher.show_processing_time()
