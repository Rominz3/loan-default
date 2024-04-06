import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

training_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Training Data.csv')
test_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Test Data.csv')
data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\combined Data.csv')

sample_size = 150
sample_data = data.sample(sample_size, random_state=42)
pair_plot = sns.pairplot(sample_data, hue='Risk_Flag', diag_kind='kde', palette='husl')

for ax in pair_plot.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8) 

plt.suptitle('Pairplot of Quantitative Features')

plt.show()

