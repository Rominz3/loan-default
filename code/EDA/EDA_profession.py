import pandas as pd 
import matplotlib.pyplot as plt

training_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Training Data.csv')
test_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Test Data.csv')
data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\combined Data.csv')

print(data['Profession'].unique())
fig, ax = plt.subplots(figsize=(100, 8))

x_labels = data['Profession'].unique()

data['Profession'].value_counts().reindex(x_labels).plot.bar(ax=ax)

ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=90, ha='right')
plt.subplots_adjust(top=0.99,bottom=0.4)
plt.show()

