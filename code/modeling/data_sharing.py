import pandas as pd


data= pd.read_csv(r'F:\job\Loan Default Prediction Analysis\scaled encoded Data.csv')


group1 = data.iloc[:252000, :]
group2 = data.iloc[252000:, :]


group1.to_csv(r'F:\job\Loan Default Prediction Analysis\training scaled encoded Data.csv', index=False)
group2.to_csv(r'F:\job\Loan Default Prediction Analysis\test scaled encoded Data.csv', index=False)