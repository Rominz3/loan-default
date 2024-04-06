import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data= pd.read_csv(r'F:\job\Loan Default Prediction Analysis\scaled Data.csv')


count_encoder_city = ce.CountEncoder()
data['city_count'] = count_encoder_city.fit_transform(data['CITY'])


count_encoder_state = ce.CountEncoder()
data['state_count'] = count_encoder_state.fit_transform(data['STATE'])


label_encoder_House_Ownership = LabelEncoder()
data['House_Ownership'] = label_encoder_House_Ownership.fit_transform(data['House_Ownership'])


label_encoder_profession_group = LabelEncoder()
data['profession_group'] = label_encoder_profession_group.fit_transform(data['profession_group'])


label_encoder_Married_Single = LabelEncoder()
data['Married/Single'] = label_encoder_Married_Single.fit_transform(data['Married/Single'])


label_encoder_Car_Ownership = LabelEncoder()
data['Car_Ownership'] = label_encoder_Car_Ownership.fit_transform(data['Car_Ownership'])

data=data.drop(['CITY','STATE'],axis=1)

data.to_csv(r'F:\job\Loan Default Prediction Analysis\scaled encoded Data.csv', index=False)