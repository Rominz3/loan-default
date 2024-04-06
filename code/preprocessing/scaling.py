from sklearn.preprocessing import StandardScaler
import pandas as pd

data= pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Data.csv')

columns_to_scale = ['Income','Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

standard_scaler = StandardScaler()

data[columns_to_scale] = standard_scaler.fit_transform(data[columns_to_scale])

data.to_csv(r'F:\job\Loan Default Prediction Analysis\scaled Data.csv', index=False)




