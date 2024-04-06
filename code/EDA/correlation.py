import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

training_data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Training Data.csv')
test_data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Test Data.csv')
data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\combined Data.csv')

correlation_matrix = data.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

plt.xticks(rotation=20, ha='right')

plt.title('Correlation Matrix')

plt.tight_layout()

plt.show()

