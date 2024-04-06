import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

training_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Training Data.csv')
test_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Test Data.csv')
data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\combined Data.csv')

plt.figure(figsize=(8, 6))
sns.countplot(x='House_Ownership', hue='Risk_Flag', data=data, palette='husl')
plt.title('Count Plot of House_Ownership with Risk_Flag')
plt.show()

House_Ownership_counts = data['House_Ownership'].value_counts()
plt.pie(House_Ownership_counts, labels=House_Ownership_counts.index, autopct='%1.1f%%')
plt.title('Distribution of House_Ownership')
plt.show()
