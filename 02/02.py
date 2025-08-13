import pandas as pd
import os

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

