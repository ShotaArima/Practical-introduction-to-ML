import pandas as pd
from sensor_def_04 import (
    check_machine_status_transition,
    plot_anomaly_list,
    add_stabilization_status,
    generate_lag_features,
    generate_diff_features,
    genereate_moving_average_features,
    select_features,
    calc_influence_score,
)
from sklearn.ensemble import IsolationForest
from model import (
    create_pipeline,
    predict,
    plot_predicted_results,
)
from sklearn.neighbors import LocalOutlierFactor

# データ取得
df = pd.read_csv('data/sensor.csv')

# timestampカラムについて
print("timestampカラムについて")

df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
df = df.set_index("timestamp")

print(df.index.min())
print(df.index.max())
print(df.index.max() - df.index.min())
print(len(df.index.drop_duplicates()))

# machine_statusカラムについて
print("machine_statusカラムについて")
print(df["machine_status"].value_counts(dropna=False))

check_machine_status_transition(df)

# sensorカラムについて
print("sensorカラムについて")
print(df.describe().loc["count"].sort_values()[:3])

print("センサデータの可視化 (s00, s01)")
plot_anomaly_list(df, s_list=["s00", "s01"])

# 異常のあるセンサデータを抽出して可視化
broken_indices = df[df['machine_status'] == 'BROKEN'].index
indices = (
    pd.DatetimeIndex([df.index[0]])
    .append(broken_indices)
    .append(pd.DatetimeIndex([df.index[-1]]))
)
print("故障点によって分割されたセンサデータの可視化 (s00):anomaly0, anomaly1, ...")
for i in range(len(indices) -1 ):
  plot_anomaly_list(
      df.loc[indices[i]:indices[i+1]],
      s_list = ["s00"],
      title = f"anomaly{i}"
  )

# 欠損値を直前の値で補完
df = df.bfill()

# 修理後センサ値が不安定
print("修理後センサ値の安定性について")
df = add_stabilization_status(df)

# データ読み込み時に以下のようにしておくことをおすすめする
df = (
    pd.read_csv(
        "data/sensor.csv",
        parse_dates = ["timestamp"],
        index_col = "timestamp"
    )
    .bfill()
    .ffill()
    .pipe(add_stabilization_status)
)

# 特徴量エンジニアリング
print("lag特徴量:故障10分前のセンサ値")
tmp_df = df[["s00"]]
print(pd.concat([
    tmp_df,
    generate_lag_features(tmp_df[["s00"]], periods=1)
], axis=1).head())

print("diff特徴量:故障10分前のセンサ値 (diff(s00, 1))")
tmp_df = df.loc[broken_indices[0]:broken_indices[1], ["s00", "machine_status"]]
plot_anomaly_list(
  pd.concat([
      tmp_df,
      generate_diff_features(tmp_df[["s00"]], periods=1)
  ], axis=1),
)

# 移動平均特徴量
tmp_df = df.loc[:broken_indices[0], ["s06", "machine_status"]]
print("移動平均特徴量:過去20分のs06の平均値 (ma(s06, 20))")
plot_anomaly_list(
    pd.concat([
        tmp_df,
        genereate_moving_average_features(tmp_df[["s06"]], window=20)
    ], axis=1)
)

print("lag移動中央値特徴量:過去20分のs06の中央値 (med(lag(s06, 20)))")
tmp_df = df.loc[:broken_indices[0], ["s06", "machine_status"]]
plot_anomaly_list(
    pd.concat([
      tmp_df,
      generate_lag_features(tmp_df[["s06"]], periods=10), # ラグ特徴量
      generate_lag_features(tmp_df[["s06"]], periods=10).pipe(
          genereate_moving_average_features,
          window = 10,
      ), # 移動平均ラグ特徴量
    ], axis=1),
)

print("diff移動平均特徴量:過去10分のs06の平均値 (ma(diff(s06, 10)))")
tmp_df = df.loc[:broken_indices[0], ["s06", "machine_status"]]
plot_anomaly_list(
    pd.concat([
        tmp_df,
        generate_diff_features(tmp_df[["s06"]], periods=10), # 差分特徴量
        generate_diff_features(tmp_df[["s06"]], periods=10).pipe(
            genereate_moving_average_features,
            window = 10,
        ), # 移動平均差分特徴量
    ], axis=1),
)

# 特徴量選択
print("0.7以上の相関を持つ特徴量を削除")
normal_df = df[df["machine_status"] == "NORMAL"].drop(columns=["machine_status"])
features_df, selected_features_params = select_features(normal_df)

# 機械学習モデル
print("Isolation Forestによる異常検知")
pipeline_isolationforest = create_pipeline(
    anomaly_detector=IsolationForest(
        random_state=1234,
    ),
    numeric_feature_names=features_df.columns.tolist()
)
pipeline_isolationforest.fit(features_df)
print("Isolation Forestの予測")
predicted_isolationforest_df = predict(
    features_df,
    predicted_func=pipeline_isolationforest.predict,
    score_func=pipeline_isolationforest.decision_function,
)
print(predicted_isolationforest_df.head())

# 可視化
print("Isolation Forestの予測結果の可視化 (anomaly: 0~7)")
plot_predicted_results(
    predicted_isolationforest_df,
    broken_indices=broken_indices,
    indices=indices,
    original_df=df,
    threshold=0.08,
    model="Isolation Forest",
)

# LOFによる異常検知
pipeline_lof = create_pipeline(
    anomaly_detector=LocalOutlierFactor(
        n_neighbors=20,
    ),
    numeric_feature_names=features_df.columns.tolist()
)
predicted_lof_df = predict(
    features_df,
    predicted_func=pipeline_lof.fit_predict,
    score_func=lambda x: pipeline_lof[1].negative_outlier_factor_,
)
print("LOFの予測 (anomaly: 0~7)")
plot_predicted_results(
    predicted_lof_df,
    original_df=df,
    threshold=-1.5,
    broken_indices=broken_indices,
    indices=indices,
    model="LOF",
)

print("モデルの評価")
# でたらめの確認
end = broken_indices[1]
begin = end - pd.Timedelta(days=1)

plot_predicted_results(
    predicted_lof_df[begin:end],
    extra_result_df=predicted_isolationforest_df[begin:end],
    original_df=df,
    score_label="Prediction Score (LOF)",
    extra_score_label="Pridiction Score (Isolation Forest)",
    broken_indices = broken_indices,
    indices = indices,
    model = "IsolationForest vs LOF",
)

for i in range(len(indices) - 1):
    # 期間内のスコアが最小の時点
    idx = predicted_isolationforest_df.loc[
        (indices[i] < predicted_isolationforest_df.index) &
        (predicted_isolationforest_df.index < indices[i + 1]),
        'score'
    ].idxmin()

    print(f"### Period {i} ###")
    print("Moment of lowest score")
    print(predicted_isolationforest_df.loc[[idx]])

    calc_influence_score(
        df=features_df.loc[[idx]],
        original_df=features_df,
        score_func=pipeline_isolationforest.decision_function,
    )
