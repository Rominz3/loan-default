import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


train_data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\training scaled encoded Data.csv')
test_data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\test scaled encoded Data.csv')


X_train = train_data.drop('Risk_Flag', axis=1)
y_train = train_data['Risk_Flag']


xgb_model = XGBClassifier(learning_rate=0.3, max_depth=7, n_estimators=200, random_state=42)


xgb_model.fit(X_train, y_train)


feature_importance = xgb_model.feature_importances_


plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance, align='center')
plt.xticks(range(len(feature_importance)), X_train.columns, rotation='vertical')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance in XGBoost')
plt.subplots_adjust(top=0.99,bottom=0.4)
plt.show()