import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils.analise_utils import show_dataframe

dataframe_exercicios = pd.read_csv('../data/workout_evolution/exercicio_musculacao.csv')
print(show_dataframe(dataframe_exercicios.head()))

dataframe_exercicios = dataframe_exercicios.rename(
    columns={
        'Data': 'data',
        'Nome do Exercício de Musculação': 'exercicio',
        'Duração do Treino (minutos)': 'duracao',
        'Peso (em KG)': 'peso',
        'Repetições': 'repeticoes'
    }
)

print('Dados após renomear colunas')
print(show_dataframe(dataframe_exercicios.head()))

print('Quantidade de dados de cada exercício')
df_count_exercicios = dataframe_exercicios['exercicio'].value_counts().reset_index()
print(show_dataframe(df_count_exercicios))

print('Somente Levantamento Terra, ordenado por Peso e Repetições acententemente')
df_exercicio = dataframe_exercicios.query('exercicio == "Levantamento Terra"')
df_exercicio = df_exercicio.sort_values(by=['data', 'peso', 'repeticoes'], ascending=True)
print(show_dataframe(df_exercicio.head(10)))

print('Somente Levantamento Terra, ordenado por Peso e Repetições acententemente e agrupado por pelo mês da data')
df_exercicio['data'] = pd.to_datetime(df_exercicio['data'])
df_exercicio_agrupado_mes = df_exercicio.groupby(df_exercicio['data'].dt.strftime('%Y-%m')).agg({'peso': 'mean'}).reset_index()
print(show_dataframe(df_exercicio_agrupado_mes))

plt.figure(figsize=(10, 6))
sns.lineplot(x='data', y='peso', data=df_exercicio_agrupado_mes, marker='o')

plt.title('Peso ao Longo do Tempo')
plt.xlabel('Mês e Ano')
plt.ylabel('Peso (kg)')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
