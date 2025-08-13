import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('data/employee_train.csv')
print(df.head())

# Features and target variable
target_col = 'leaving'
feature_cols = ['age', 'job_satisfaction', 'environment_satisfaction', 'over_time', 'monthly_income']

y_train = df[target_col]
X_train = df[feature_cols]

model = LogisticRegression()
model.fit(X_train, y_train)

df["pred_leaving"] = model.predict(X_train)
# print(df[['leaving', 'pred_leaving']])
accuracy = accuracy_score(df['leaving'], df['pred_leaving'])
print("Accuracy:", accuracy)

df_pred = pd.read_csv('data/employee_pred.csv')
feature_cols = ['age', 'job_satisfaction', 'environment_satisfaction','over_time', 'monthly_income']
X_pred = df_pred[feature_cols]
X_pred['pred_leaving'] = model.predict(X_pred)
print(X_pred.head())
