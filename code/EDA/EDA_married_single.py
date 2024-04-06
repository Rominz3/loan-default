import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

training_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Training Data.csv')
test_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Test Data.csv')
data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\combined Data.csv')

plt.figure(figsize=(8, 6))
sns.countplot(x='Married/Single', hue='Risk_Flag', data=data, palette='husl')
plt.title('Count Plot of Married/Single with Risk_Flag')
plt.show()

Married_Single_counts = data['Married/Single'].value_counts()
plt.pie(Married_Single_counts, labels=Married_Single_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Married/Single')
plt.show()
