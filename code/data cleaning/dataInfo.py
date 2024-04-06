import pandas as pd
training_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Training Data.csv')
test_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Test Data.csv')
data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\combined Data.csv')
print(data.describe())
print(data.info())
print(pd.isnull(data).sum())
print(training_data.duplicated().sum())
print(test_data.duplicated().sum())