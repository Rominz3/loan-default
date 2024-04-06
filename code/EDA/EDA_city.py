import pandas as pd 
import matplotlib.pyplot as plt

training_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Training Data.csv')
test_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Test Data.csv')
data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\combined Data.csv')

unique_cities = data['CITY'].str.split(',').explode().unique()

print(', '.join(unique_cities))

