import pandas as pd
from tabulate import tabulate

dataframe_exercicios = pd.read_csv('../data/workout_evolution/exercicio_musculacao.csv')
print(tabulate(dataframe_exercicios.head(1), headers='keys', tablefmt='grid', stralign='left', showindex=False))

print()

dataframe_exercicios2 = pd.read_csv('../data/workout_evolution/strong.csv')
print(tabulate(dataframe_exercicios2.head(1), headers='keys', tablefmt='grid', stralign='left', showindex=False))

print()

dataframe_exercicios3 = pd.read_csv('../data/workout_evolution/weightlifting_721_workouts.csv')
print(tabulate(dataframe_exercicios3.head(1), headers='keys', tablefmt='grid', stralign='left', showindex=False))