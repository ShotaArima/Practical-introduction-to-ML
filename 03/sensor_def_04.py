import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from typing import Optional

def check_machine_status_transition(df: pd.DataFrame):
  def print_duration(current, last_change_at, status):
    print(f"{current}: {status:10s} lasted {current - last_change_at}")

  status = df["machine_status"][0]
  last_change_at = df.index[0]
  for i, rows in df.iterrows():
    if status != rows["machine_status"]:
      print_duration(i, last_change_at, status)
      last_change_at = i
    status = rows["machine_status"]
  print_duration(i, last_change_at, status)


def mask_and_other_is_nan(sr, cond, mask_value):
  return sr.mask(cond, mask_value).where(lambda sr :sr == mask_value)

def get_plot_series(df, s):
  machine_status = df["machine_status"]
  sensor = df[s]
  s_max = sensor.max()
  s_min = sensor.min()
  s_height = s_max - s_min
  min_y = s_min - s_height*0.1
  max_y = s_max - s_height*0.1

  broken = mask_and_other_is_nan(
      machine_status,
      machine_status == "BROKEN",
      min_y
  )
  normal = mask_and_other_is_nan(
      machine_status,
      machine_status == "NORMAL",
      max_y
  )
  return normal, broken, sensor, min_y, max_y

def detect_status_regions(df, s: Optional[str] = None):
    current = None
    begin = 0
    regions = []
    for i, status in enumerate(df['machine_status']):
        if s is not None and pd.isnull(df[s].iloc[i]):
            status = "Missing"

        if status == current:
            continue
        else:
            if current is not None:
                regions.append({"status": current, "begin": begin, "end": i})
            current = status
            begin = i
    else:
        regions.append({"status": current, "begin": begin, "end": i})
    return regions

def plot_anomaly(df: pd.DataFrame, s: str, title: Optional[str] = None,):
    normal, broken, sensor, min_y, max_y = get_plot_series(df, s)

    plt.figure(figsize=(16, 2))

    # センサーの値のプロット
    plt.plot(sensor, linewidth=0.5, label=s)
    # machine_status (BROKEN以外)のプロット
    regions = detect_status_regions(df, s)
    colors = {
        "NORMAL": "#e7f5fc",
        "RECOVERING": "#e2e3e4",
        "Missing": "#b2b3b6",
        "STABILIZATION": "#95d8f5",
    }
    plotted = set()
    for region in regions:
        if region['status'] == "BROKEN":
            continue
        plt.fill_between(
            df.iloc[region['begin']: region['end']].index,
            min_y,
            max_y,
            alpha = 0.5,
            color = colors[region['status']],
            label = region['status'] if region['status'] not in plotted else "",
        )
        plotted.add(region['status'])
    # BROKEN ラベルが付与された点のプロット
    plt.plot(
        broken, linestyle='none', marker='X', label="broken", color='black'
    )

    plt.title(title or s)
    plt.legend()
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig(f'output/04/{title or s}.png')


def plot_anomaly_list(
        df: pd.DataFrame,
        s_list: list[str] = None,
        title: Optional[str] = None,
):
    if s_list is None:
        s_list = df.describe().columns.tolist()
    for s in s_list:
        plot_anomaly(df, s, title)

def add_stabilization_status(df):
    # RECOVERING から NORMAL に変わるポイント
    recovering_to_normal = (
            (df["machine_status"] == "NORMAL")
            & (df["machine_status"].shift(1) == "RECOVERING")
    )
    recovering_to_normal_index = df.index[recovering_to_normal]

    # NORMAL から将来のx分間のs00のstdが初めて0.05未満かつmeanが2.2以上になるポイントを見つけ、それまでをSTABILIZATIONとする
    for i in range(len(recovering_to_normal_index)):
        idx = recovering_to_normal_index[i]
        print(idx)
        minutes = 0
        add_minutes = 60
        window_size = 360
        limit = (
            recovering_to_normal_index[i+1]
            if i != len(recovering_to_normal_index) -1
            else df.index.max()
        )
        while(True):
            begin = idx + pd.Timedelta(minutes = minutes)
            end = idx + pd.Timedelta(minutes = minutes+window_size)
            std = df.loc[begin:end, "s00"].std()
            mean = df.loc[begin:end, "s00"].mean()
            if std < 0.05 and 2.2 <= mean:
                df.loc[idx:begin, "machine_status"] = "STABILIZATION"
                break
            minutes += add_minutes

            if limit < end:
                df.loc[idx:end, "machine_status"] = "STABILIZATION"
                break
    return df

def generate_lag_features(df, periods):
    return df.shift(periods=periods).bfill().rename(
        columns={col: f"lag({col}, {periods})"for col in df.columns}
    )

def generate_diff_features(df, periods):
    return df.diff(periods=periods).bfill().rename(
        columns={col: f"diff({col}, {periods})"for col in df.columns}
    )

def genereate_moving_average_features(df, window):
    return df.rolling(window=window).mean().bfill().rename(
        columns={col: f"ma({col}, {window})" for col in df.columns}
    )

def genereate_moving_median_features(df, window):
    return df.rolling(window=window).median().bfill().rename(
        columns={col: f"med({col}, {window})" for col in df.columns}
    )

def generate_features(df, perod, window, use_median, use_lag):
    new_df = df.copy()
    if perod != None:
        if use_lag:
            func = generate_lag_features
        else:
            func = generate_diff_features
        new_df = new_df.pipe(func, periods=perod)
    if window != None:
        if use_median:
            func = genereate_moving_median_features
        else:
            func = genereate_moving_average_features
        new_df = new_df.pipe(func, window=window)

    return new_df

def drop_highly_correlated_columns(df):
    melted_corr = (
        df[df.describe().columns]
        .corr()
        .where(lambda df: np.triu(np.ones(df.shape), k=1).astype(bool))  # 上三角部分のみ残す
        .reset_index()
        .melt(id_vars="index")
        .where(lambda df: df["index"] != df["variable"])
        .where(lambda df: df["value"] > 0.7)
        .dropna()
        .sort_values(["index", "variable"])
    )
    drop_columns = sorted(list(set(melted_corr["variable"])))
    return df.drop(columns=drop_columns)

def select_features(df: pd.DataFrame):
    periods = [None, 1, 3, 6, 12, 18]
    use_lags = [True, False]
    windows = [None, 1, 3, 6, 12, 18]
    use_medians = [True]

    result_df = df.copy()
    selected_feature_params = []
    for original_col in df.filter(regex="^s\d\d").columns:
        print(original_col)
        new_dfs = []
        for period, window, use_lag in product(
            periods, windows, use_lags,
        ):
            if period is None and window is None:
                continue

            if period is None and use_lags:
                continue

            sensors_df = df[[original_col]]
            new_df = generate_features(
                sensors_df, period, window,
                use_median=True, use_lag=use_lag
            )
            new_dfs.append(new_df)

        features_df = pd.concat(new_dfs, axis=1)
        print(features_df.shape)
        features_df = drop_highly_correlated_columns(features_df)
        print(features_df.shape)

        result_df = pd.concat(
            [result_df, features_df],
            axis=1
        )
        print()

    print(f"selected: {result_df.columns}")
    return result_df, selected_feature_params


def calc_influence_score(df, original_df, score_func):
    score = score_func(df)

    assert len(df) == 1

    result_df = pd.DataFrame(index=df.describe().columns.tolist())
    new_scores = []
    np.random.seed(1234)
    for col in df.describe().columns:
        u_limit = original_df[col].max()
        l_limit = original_df[col].min()
        random_values = np.random.uniform(l_limit, u_limit, 100)
        new_df = pd.concat([df] * 100, ignore_index=True)
        new_df[col] = random_values
        new_scores.append(score_func(new_df).mean())

    result_df["score"] = new_scores - score
    result_df = result_df.sort_values("score")

    print("5 features with the largest negative influence at the moment")
    print(result_df.head())

    print("5 features with the largest positive influence at the moment")
    print(result_df.tail())
