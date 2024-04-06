import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


train_data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\training scaled encoded Data.csv')
test_data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\test scaled encoded Data.csv')


X_train = train_data.drop('Risk_Flag', axis=1)
y_train = train_data['Risk_Flag']


X_test = test_data.drop('Risk_Flag', axis=1)
y_test = test_data['Risk_Flag']


param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    # 'subsample': [0.8, 0.9, 1.0],
    # 'colsample_bytree': [0.8, 0.9, 1.0]
}


xgb_model = XGBClassifier(random_state=42)


grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='accuracy')


y_test_pred_probs = best_model.predict_proba(X_test)[:, 1]
test_accuracy = accuracy_score(y_test, best_model.predict(X_test))



print(f'Best Hyperparameters: {best_params}')
print(f'Cross Validation Scores: {cv_scores}')
print(f'Average Cross Validation Accuracy: {cv_scores.mean()}')
print(f'Accuracy on Test Data with Best Model: {test_accuracy}')

