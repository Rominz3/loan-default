import category_encoders as ce
import pandas as pd

data= pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Data.csv')


count_encoder_city = ce.CountEncoder()
data['city_count'] = count_encoder_city.fit_transform(data['CITY'])

count_encoder_state = ce.CountEncoder()
data['state_count'] = count_encoder_state.fit_transform(data['STATE'])


data.to_csv(r'F:\job\Loan Default Prediction Analysis\Data.csv', index=False)
