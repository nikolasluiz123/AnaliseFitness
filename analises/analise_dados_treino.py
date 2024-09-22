import seaborn as sns
from matplotlib import pyplot as plt

from straction.normalize_training_data import get_dataframe_training_data
from utils.analise_utils import show_dataframe

df_training = get_dataframe_training_data()

show_dataframe(df_training.head())

df_training.info()

# Correlação entre série, peso e repetições
correlation = df_training[['serie', 'peso', 'repeticoes']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()

# Frequência dos pesos
df_training['peso'].hist(bins=50)
plt.title('Distribuição de pesos')
plt.xlabel('Peso')
plt.ylabel('Frequência')
plt.show()
