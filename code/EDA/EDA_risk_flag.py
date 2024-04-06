import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

training_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Training Data.csv')
test_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Test Data.csv')
data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\combined Data.csv')

plt.figure(figsize=(8, 6))
sns.countplot(x='Risk_Flag', hue='Risk_Flag', data=data, palette='husl')
plt.title('Count Plot of Risk_Flag')
plt.show()

Risk_Flag_counts = data['Risk_Flag'].value_counts()
plt.pie(Risk_Flag_counts, labels=Risk_Flag_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Risk_Flag')
plt.show()
