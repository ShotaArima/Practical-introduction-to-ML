import os
import matplotlib
matplotlib.use("Agg")  # GUIのない環境なら推奨
import matplotlib.pyplot as plt
import pandas as pd
from darts.timeseries import TimeSeries
from darts.models import ExponentialSmoothing

df = pd.read_csv('data/visitors.csv')
series = TimeSeries.from_dataframe(df, time_col='date', value_cols=['visitors'])

ax = series.plot()
fig = ax.get_figure()
os.makedirs("out", exist_ok=True)
fig.savefig("output/02_5month_visitors.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# モデルの作成
model = ExponentialSmoothing()
model.fit(series)

predictions = model.predict(36)
ax = series[-72:].plot()
fig = ax.get_figure()
os.makedirs("out", exist_ok=True)
fig.savefig("output/02_2month-harf_visitors.png", dpi=300, bbox_inches="tight")
plt.close(fig)

ax = predictions.plot(label='forecast')
fig = ax.get_figure()
os.makedirs("out", exist_ok=True)
fig.savefig("output/02_predict.png", dpi=300, bbox_inches="tight")
plt.close(fig)

train, test = series[:-36], series[-36:]
# train.plot(label='train')
# test.plot(label='test')

fig, ax = plt.subplots(figsize=(9, 4))

train.plot(ax=ax, label="train")
test.plot(ax=ax, label="test")

# スプリット位置に縦線（任意）
try:
    ax.axvline(test.start_time(), linestyle="--", linewidth=1)
except Exception:
    pass

ax.legend()
ax.set_title("Train/Test Split")
fig.savefig("output/02_train_test.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 学習
model = ExponentialSmoothing()
model.fit(train)

# 予測
prediction = model.predict(36)

# 同じ図に重ねて描く → 保存
fig, ax = plt.subplots(figsize=(9, 4))

series[-72:].plot(ax=ax, label="history")   # () を忘れない！
prediction.plot(ax=ax, label="forecast")    # 同じ ax に描く

ax.legend()
ax.set_title("History + Forecast (ExponentialSmoothing)")
fig.savefig("output/02_history_forecast.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 予測も追加
pred = model.predict(36)
fig, ax = plt.subplots(figsize=(9, 4))
series[-72:].plot(ax=ax)
pred.plot(ax=ax, label="forecast")
ax.legend()
fig.savefig("output/02_train_test_forecast.png", dpi=300, bbox_inches="tight")
plt.close(fig)
