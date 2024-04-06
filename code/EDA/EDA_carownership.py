import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

training_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Training Data.csv')
test_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Test Data.csv')
data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\combined Data.csv')

plt.figure(figsize=(8, 6))
sns.countplot(x='Car_Ownership', hue='Risk_Flag', data=data, palette='husl')
plt.title('Count Plot of Car_Ownership with Risk_Flag')
plt.show()

Car_Ownership_counts = data['Car_Ownership'].value_counts()
plt.pie(Car_Ownership_counts, labels=Car_Ownership_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Car_Ownership')
plt.show()
