import pandas as pd

from utils.analise_utils import show_dataframe

dataframe_exercicios_1 = pd.read_csv('../data/workout_evolution/strong.csv')
dataframe_exercicios_1 = dataframe_exercicios_1.drop(columns=['Distance', 'Seconds', 'Notes', 'Workout Notes', 'RPE', 'Duration'])
dataframe_exercicios_1 = dataframe_exercicios_1.rename(
    columns={
        'Date': 'data',
        'Workout Name': 'treino',
        'Exercise Name': 'exercicio',
        'Set Order': 'serie',
        'Weight': 'peso',
        'Reps': 'repeticoes'
    }
)

print(show_dataframe(dataframe_exercicios_1.head()))

print(f'Shape dados csv 1: {dataframe_exercicios_1.shape}')

print('Quantidade de dados de cada exercício')
df_count_exercicios = dataframe_exercicios_1['exercicio'].value_counts().reset_index()
print(show_dataframe(df_count_exercicios))

print('Somente Bench Press (Barbell), ordenado por Peso e Repetições acententemente e agrupado por pelo mês da data')
df_exercicio = dataframe_exercicios_1.query('exercicio == "Bench Press (Barbell)"')
df_exercicio = df_exercicio.sort_values(by=['data', 'peso', 'repeticoes'], ascending=True)

dataframe_exercicios_1['data'] = pd.to_datetime(dataframe_exercicios_1['data'])
df_exercicio_agrupado_mes = dataframe_exercicios_1.groupby(dataframe_exercicios_1['data'].dt.strftime('%Y-%m')).agg({'peso': 'mean'}).reset_index()

print(show_dataframe(df_exercicio_agrupado_mes))

print('Treino do dia 2021-09-13 10:41:41 ')
data_especifica = '2021-09-13 10:41:41'
df_treino_dia = dataframe_exercicios_1.query('data == @data_especifica')
print(show_dataframe(df_treino_dia))
df_treino_dia.to_csv('../data/workout_evolution/df_treino_dia', index=False)

print('---------------------------------------------------------------------------------------------------------------')

dataframe_exercicios_2 = pd.read_csv('../data/workout_evolution/weightlifting_721_workouts.csv')
dataframe_exercicios_2 = dataframe_exercicios_2.drop(columns=['Distance', 'Seconds', 'Notes', 'Workout Notes'])
dataframe_exercicios_2['Weight'] = dataframe_exercicios_2['Weight'] * 0.453592
dataframe_exercicios_2 = dataframe_exercicios_2.rename(
    columns={
        'Date': 'data',
        'Workout Name': 'treino',
        'Exercise Name': 'exercicio',
        'Set Order': 'serie',
        'Weight': 'peso',
        'Reps': 'repeticoes'
    }
)
dataframe_exercicios_2['data'] = pd.to_datetime(dataframe_exercicios_2['data'])

print(show_dataframe(dataframe_exercicios_2.head()))

print(f'Shape dados csv 2: {dataframe_exercicios_2.shape}')

print('Quantidade de dados de cada exercício')
df_count_exercicios = dataframe_exercicios_2['exercicio'].value_counts().reset_index()
print(show_dataframe(df_count_exercicios))

print('Somente Bench Press (Barbell), ordenado por Peso e Repetições acententemente e agrupado por pelo mês da data')
df_exercicio2 = dataframe_exercicios_2.query('exercicio == "Bench Press (Barbell)"')
df_exercicio2 = df_exercicio2.sort_values(by=['data', 'peso', 'repeticoes'], ascending=True)

dataframe_exercicios_2['data'] = pd.to_datetime(dataframe_exercicios_2['data'])
df_exercicio_agrupado_mes2 = dataframe_exercicios_2.groupby(dataframe_exercicios_2['data'].dt.strftime('%Y-%m')).agg({'peso': 'mean'}).reset_index()

print(show_dataframe(df_exercicio_agrupado_mes2))

print('Treino do dia 2021-09-13 10:41:41 ')
data_especifica = '2015-10-23 17:06:37'
df_treino_dia = dataframe_exercicios_2.query('data == @data_especifica')
print(show_dataframe(df_treino_dia))

