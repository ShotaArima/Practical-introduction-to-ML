import pandas as pd
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# データの読み取り
path = 'data/realestate_train.csv'

if path is not None:
    print(f"{path} is not None")
    print("Current working directory:", os.getcwd())
    df = pd.read_csv(path)  # ← ここで読み込む
    print(df.head())
else:
    df = pd.read_csv('02/data/realestate_train.csv')
    print(df.head())

# 特徴量と正解データの設定

target_col = "rent_price"
feature_cols = ['house_area', 'distance']

y = df[target_col]
X = df[feature_cols]

# モデルの学習
model = Ridge()
model.fit(X, y)

# 予測
df['pred_rent_price'] = model.predict(X)
print(df[['rent_price', 'pred_rent_price']])

# 評価
print("MAE:", mean_absolute_error(df[target_col], df['pred_rent_price']))


# 高い精度のモデルができたとした際の汎化性能について

# Load the dataset
df_pred = pd.read_csv('data/realestate_pred.csv')
print(df_pred.head())

featrure_cols = ['house_area', 'distance']
X_pred = df_pred[featrure_cols]
X_pred['pred_rent_price'] = model.predict(X_pred)

print(X_pred.head())
