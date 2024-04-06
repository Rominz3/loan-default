import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\training scaled encoded Data.csv')
test_data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\test scaled encoded Data.csv')

X_train = train_data.drop('Risk_Flag', axis=1)
y_train = train_data['Risk_Flag']

X_test = test_data.drop('Risk_Flag', axis=1)
y_test = test_data['Risk_Flag']

dt_model = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf=4, min_samples_split=2, random_state=42)


cv_scores = cross_val_score(dt_model, X_train, y_train, cv=3, scoring='accuracy')
print(f'Cross Validation Scores: {cv_scores}')
print(f'Average Cross Validation Accuracy: {cv_scores.mean()}')


dt_model.fit(X_train, y_train)


y_test_pred = dt_model.predict(X_test)


test_accuracy = accuracy_score(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred)
print(f'Accuracy on Test Data: {test_accuracy}')
print('Classification Report on Test Data:')
print(test_report)


conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print('Confusion Matrix on Test Data:')
print(conf_matrix_test)



plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix on Test Data')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
