import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class CrossValScoreResult:
    """
        Classe para armazenar os valores das métricas matemáticas que podem ser analisadas e auxiliar no julgamento
        do treinamento do modelo.
    """

    def __init__(self,
                 mean: float,
                 standard_deviation: float,
                 median: float,
                 variance: float,
                 standard_error: float,
                 min_max_score: tuple[float, float],
                 estimator):
        """
            :param mean: Média dos scores individuais, fornece uma estimativa central do desempenho do modelo.
            :param standard_deviation: Desvio Padrão, mede a variação dos scores em diferentes folds. Um Desvio Padrão
            baixo indica que o modelo tem desempenho consistente, enquanto um desvio padrão alto indica variabilidade
            entre os folds.
            :param median: A mediana dos scores é uma métrica robusta que representa o valor central da distribuição dos
            scores, sendo menos sensível a outliers.
            :param variance: A variância mede a dispersão dos scores e está relacionada ao desvio padrão, sendo o
            quadrado deste.
            :param standard_error: O erro padrão da média estima a precisão da média dos scores, mostrando o quão longe
            a média estimada está da média verdadeira.
            :param min_max_score: O score máximo e mínimo ajudam a identificar a melhor e a pior performance entre os
            folds.
            :param estimator Estimador com os melhores parâmetros e que foi testado.
        """

        self.mean = mean
        self.standard_deviation = standard_deviation
        self.median = median
        self.variance = variance
        self.standard_error = standard_error
        self.min_max_score = min_max_score
        self.estimator = estimator

    def show_cross_val_metrics(self):
        """
        Função para exibir as métricas de validação cruzada de forma clara e estruturada.
        """

        print("Resultados das Métricas de Validação Cruzada")
        print("-" * 50)
        print(f"Média dos scores          : {self.mean:.4f}")
        print(f"Desvio padrão             : {self.standard_deviation:.4f}")
        print(f"Mediana dos scores        : {self.median:.4f}")
        print(f"Variância dos scores      : {self.variance:.4f}")
        print(f"Erro padrão da média      : {self.standard_error:.4f}")
        print(f"Score mínimo              : {self.min_max_score[0]:.4f}")
        print(f"Score máximo              : {self.min_max_score[1]:.4f}")
        print(f"Melhor Estimator          : {self.estimator} ")
        print("-" * 50)
        print("Análise:")
        print(f"- Um desvio padrão baixo ({self.standard_deviation:.4f}) indica consistência no desempenho.")
        print(f"- A variância ({self.variance:.4f}) reflete o grau de dispersão dos scores.")
        print(f"- O erro padrão ({self.standard_error:.4f}) mostra a precisão da média calculada.")
        print(f"- O intervalo entre o score mínimo ({self.min_max_score[0]:.4f}) e máximo ({self.min_max_score[1]:.4f})"
              " indica a variabilidade entre as partições.")


class HipperParamsSearch:

    def __init__(self, data_x, data_y, params: dict[str, list], cv, seed: int = 1, n_jobs: int = -1):
        """
            :param data_x: Features obtidas dos dados analisados
            :param data_y: Classes ou o resultado que deseja obter
            :param params: Hiper parâmetros que deseja testar
            :param cv: Estratégia de divisão dos grupos
        """

        self.params = params
        self.cv = cv
        self.data_x = data_x
        self.data_y = data_y
        self.n_jobs = n_jobs

        self.start_search_parameter_time = 0
        self.end_search_parameter_time = 0

        self.start_best_model_cross_validation = 0
        self.end_best_model_cross_validation = 0

        np.random.seed(seed)

    def _search_hipper_parameters(self, number_iterations: int, estimator) -> RandomizedSearchCV:
        """
            Função para realizar a pesquisa dos melhores parâmetros para o estimador RandomForestRegressor.

            :param number_iterations: Quantidade de vezes que o RandomizedSearchCV vai escolher os valores dos parâmetros
            fornecidos no parâmetro params
            :param estimator Instância do estimador

            :return: Retorna o objeto RandomizedSearchCV que poderá ser utilizado na função calculate_cross_val_score
            e obter as métricas.
        """

        search = RandomizedSearchCV(estimator=estimator,
                                    param_distributions=self.params,
                                    cv=self.cv,
                                    n_jobs=self.n_jobs,
                                    verbose=2,
                                    n_iter=number_iterations)

        self.start_search_parameter_time = time.time()

        search.fit(X=self.data_x, y=self.data_y)

        self.end_search_parameter_time = time.time()

        return search

    def calculate_cross_val_score(self, searcher: RandomizedSearchCV):
        """
            Função para realizar a validação cruzada dos dados utilizando o resultado da busca de hiperparâmetros
            para validar o estimador encontrado.

            :return: Retorna um objeto CrossValScoreResult contendo as métricas matemáticas
        """

        self.start_best_model_cross_validation = time.time()

        scores = cross_val_score(estimator=searcher,
                                 X=self.data_x,
                                 y=self.data_y,
                                 cv=self.cv,
                                 n_jobs=self.n_jobs,
                                 verbose=2)

        self.end_best_model_cross_validation = time.time()

        return CrossValScoreResult(
            mean=np.mean(scores),
            standard_deviation=np.std(scores),
            median=np.median(scores),
            variance=np.var(scores),
            standard_error=np.std(scores) / np.sqrt(len(scores)),
            min_max_score=(np.min(scores), np.max(scores)),
            estimator=searcher.best_estimator_
        )

    def show_processing_time(self):
        """
        Função para exibir os tempos de execução de cada etapa do processo de busca de hiperparâmetros e validação cruzada
        no formato HH:MM:SS.
        """
        search_time = self.end_search_parameter_time - self.start_search_parameter_time
        validation_time = self.end_best_model_cross_validation - self.start_best_model_cross_validation

        print("Tempos de Execução")
        print("-" * 40)
        print(f"Tempo para busca de hiperparâmetros  : {self.__format_time(search_time)}")
        print(f"Tempo para validação cruzada          : {self.__format_time(validation_time)}")
        print("-" * 40)

    @staticmethod
    def __format_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


class RandomForestRegressorSearch(HipperParamsSearch):
    """
        Classe para realizar a pesquisa de hiper parametros do estimador RandomForestRegressor utilizando RandomizedSearchCV
        para evitar percorrer todas as opções de valores.
    """

    def __init__(self, data_x, data_y, params: dict[str, list], cv, seed: int = 1, n_jobs: int = -1):
        super().__init__(data_x, data_y, params, cv, seed, n_jobs)

    def search_hipper_parameters(self, number_iterations: int) -> RandomizedSearchCV:
        return super()._search_hipper_parameters(number_iterations=number_iterations, estimator=RandomForestRegressor())


class DecisionTreeRegressorSearch(HipperParamsSearch):
    """
        Classe para realizar a pesquisa de hiper parametros do estimador DecisionTreeRegressor utilizando RandomizedSearchCV
        para evitar percorrer todas as opções de valores.
    """

    def search_hipper_parameters(self, number_iterations: int) -> RandomizedSearchCV:
        return super()._search_hipper_parameters(number_iterations=number_iterations, estimator=DecisionTreeRegressor())


class SVRSearch(HipperParamsSearch):
    """
        Classe para realizar a pesquisa de hiper parametros do estimador SVR utilizando RandomizedSearchCV
        para evitar percorrer todas as opções de valores.
    """

    def search_hipper_parameters(self, number_iterations: int) -> RandomizedSearchCV:
        return super()._search_hipper_parameters(number_iterations=number_iterations, estimator=SVR())
