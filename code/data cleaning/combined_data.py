import pandas as pd

train_data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Training Data.csv')
test_data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Test Data.csv')

combined_data = pd.concat([train_data, test_data], axis=0)

combined_data.to_csv(r'F:\job\Loan Default Prediction Analysis\combined Data.csv', index=False)

