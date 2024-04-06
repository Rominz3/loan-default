import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
training_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Training Data.csv')
test_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Test Data.csv')
data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\combined Data.csv')

sample_size = 2000
sample_data = data.sample(sample_size, random_state=42)

sns.histplot(x=sample_data['CURRENT_HOUSE_YRS'], hue=sample_data['Risk_Flag'], data=data, kde=True, multiple='stack', palette='Set2')
plt.title('Histogram of CURRENT_HOUSE_YRS with Risk Flag')
plt.xlabel('CURRENT_HOUSE_YRS')
plt.ylabel('Frequency')
plt.show()

sns.boxplot(x='CURRENT_HOUSE_YRS', data=data, palette='Set3')
plt.title('Box Plot of CURRENT_HOUSE_YRS')
plt.xlabel('Risk Flag')
plt.ylabel('CURRENT_HOUSE_YRS')
plt.show()

