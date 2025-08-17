import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/wrime.tsv', sep='\\t')
# print(df.columns)

df = df[['Sentence', 'Avg. Readers_Joy', 'Avg. Readers_Sadness' ]]
# print(df.head())

# 前処理
df = df[(df['Avg. Readers_Joy'] > 0) | (df['Avg. Readers_Sadness'] > 0)]
df['JoySadness'] = df['Avg. Readers_Joy'] - df['Avg. Readers_Sadness']
df['PosiNega'] = np.where(df['JoySadness'] > 0, 'ポジティブ', 'ネガティブ')
df = df[['Sentence','PosiNega']]
df = df.head(1000)

# 文章ベクトルに変換
nlp = spacy.load('ja_ginza')
vectors = []
for _, sentence in df['Sentence'].items():
    doc = nlp(sentence)
    vectors.append(doc.vector)

X = vectors
y = df['PosiNega']

# モデル学習評価・データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

estimator = RandomForestClassifier()
estimator.fit(X_train, y_train)

print(estimator.score(X_train, y_train))
print(estimator.score(X_test, y_test))

# 予測
input_data = "体調が悪い"
doc = nlp(input_data)
y_pred = estimator.predict([doc.vector])[0]
print(f"{input_data} => {y_pred}です。")

input_data = "この商品はおすすめです。"
doc = nlp(input_data)
y_pred = estimator.predict([doc.vector])[0]
print(f"{input_data} => {y_pred}です。")
