from pandas import DataFrame
from tabulate import tabulate


def show_dataframe(dataframe: DataFrame):
    print(tabulate(dataframe, headers='keys', tablefmt='grid', stralign='left', showindex=False))
