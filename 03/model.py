from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import pandas as pd
from typing import Optional
from matplotlib import pyplot as plt
from sensor_def_04 import (
    detect_status_regions,
    mask_and_other_is_nan,
)
from sklearn.neighbors import LocalOutlierFactor

def create_preprocessor(numeric_feature_names: list[str]):
  numeric_transformer = Pipeline(steps=[
      ('scaler', StandardScaler())
  ])
  preprocessor = ColumnTransformer(
      transformers = [
          ('num', numeric_transformer, numeric_feature_names),
      ]
  )
  return preprocessor

def create_pipeline(anomaly_detector, numeric_feature_names: list[str]):
  pipeline = Pipeline([
      ('preprocessor', create_preprocessor(
          numeric_feature_names=numeric_feature_names
      )),
      ('anomaly_detector', anomaly_detector),
  ])
  return pipeline

# 予測
def predict(df, predicted_func, score_func):
  predictions_df = pd.DataFrame(
      predicted_func(df),
      index = df.index,
      columns = ["predictec"]
  )
  scores_df = pd.DataFrame(
      score_func(df),
      index = df.index,
      columns = ["score"]
  )
  result_df = (
      pd.DataFrame(
          index=pd.date_range(
              start=df.index.min(), end=df.index.max(), freq="10T"
          )
      )
      .join(predictions_df)
      .join(scores_df)
  )
  return result_df

def _regard_sequential_anomaries_as_one_anomaly(predicted, interval = 0):
    predicted.name = "predicted"
    reset_predicted = predicted.reset_index()
    new_predicted = pd.Series(
        name="predicted", index=predicted.index, dtype="float64"
    )
    for i, row in reset_predicted.iterrows():
      if (
          row["predicted"] == -1 and
          -1 in reset_predicted.iloc[
              max(0, i - interval - 1):i
          ]["predicted"].tolist()
      ):
        new_predicted.iloc[i] = 1
      else:
        new_predicted.iloc[i] = row["predicted"]

    return new_predicted

def _rescaling(sr, min_v, max_v):
    sr = (sr - sr.min()) / (sr.max() - sr.min())
    sr = sr * (max_v - min_v) + min_v
    return sr

def plot_predicted_result(
    df: pd.DataFrame, original_df: pd.DataFrame,
    extra_df: Optional[pd.DataFrame] = None,
    sensor_id: Optional[str] = None,
    threshold: Optional[float] = None,
    title: Optional[str] = None,
    score_label: Optional[str] = None,
    extra_score_label: Optional[str] = None,
    ANOMALY_LABEL = -1,
    NORMAL_LABEL = -1
):
    df = (
        df.join(original_df["machine_status"])
    )
    if sensor_id:
        df = df.join(original_df[sensor_id])

    if threshold is None:
        threshold = df["score"].quantile(
            1 - 10/(24*60/10)  # 1日に10回程度
        )
    print(f"threshold: {threshold}")

    predicted = df["score"].apply(
        lambda x: -1 if x < threshold else 1
    ).pipe(
        _regard_sequential_anomaries_as_one_anomaly,
        interval=6,  # 1時間以内に再度起こった検知は同一検知と見なす
    )
    machine_status = df["machine_status"]
    score = df["score"]

    min_v = score.min()
    max_v = score.max()
    anomaly_y = min_v - (max_v - min_v) * 0.1
    normal_y = max_v + (max_v - min_v) * 0.1

    plt.figure(figsize=(16, 2))

    regions = detect_status_regions(df)

    # machine_status のプロット
    colors = {
        "NORMAL": "#e7f5fc",
        "RECOVERING": "#e2e3e4",
        "MISSING": "#b2b3b6",
        "STABILIZATION": "#95d8f5",
    }
    for region in regions:
        if region["status"] == "BROKEN":
            continue

        plt.fill_between(
            df.iloc[region["begin"]:region["end"]].index,
            anomaly_y,
            normal_y,
            color=colors[region["status"]],
            label=region["status"],
        )

    # センサーデータのプロット
    if sensor_id:
        sensor = df[sensor_id]
        sensor = _rescaling(df[sensor_id], score.min(), score.max())
        plt.plot(sensor.ffill(),
                 linewidth=0.5, label=sensor_id)

    # 予測スコアのプロット
    plt.plot(score.ffill(),
             linewidth=0.5, label=score_label or "prediction score")

    if extra_df is not None:
        # 予測スコア(追加分)のプロット
        extra_score = _rescaling(extra_df["score"], score.min(), score.max())
        plt.plot(
            extra_score.ffill(),
            linestyle="--", linewidth=1.5,
            label=extra_score_label or "prediction score (extra)",
            color="darkgray"
        )
    else:
        # 異常と予測した点のプロット
        predicted_as_anomaly = mask_and_other_is_nan(
            predicted,
            predicted == ANOMALY_LABEL,
            anomaly_y
        )
        plt.plot(
            predicted_as_anomaly, linestyle="none", marker="x",
            label="predicted as anomaly", color="black",
        )

        # 正常と予測した点のプロット
        predicted_as_normal = mask_and_other_is_nan(
            predicted,
            predicted == NORMAL_LABEL,
            normal_y,
        )
        plt.plot(
            predicted_as_normal, linestyle="none", marker=".",
            markersize=1, label="predicted as normal",
            color="darkgray",
        )

        # BROKEN ラベルが付与された点のプロット
        actually_broken = mask_and_other_is_nan(
            machine_status,
            machine_status == "BROKEN",
            anomaly_y,
        )
        plt.plot(
            actually_broken, linestyle="none", marker="X",
            label="broken", color="black",
        )

        n_predicted_as_anomaly = len(predicted_as_anomaly.dropna())
        days = (df.index.max() - df.index.min()).total_seconds() / (60 * 60 * 24)
        print(
            f"Predicted {n_predicted_as_anomaly} anomalies "
            f"({n_predicted_as_anomaly / days:.1f} anomalies / day)"
        )
        print(
            "Last 5 points predicted as anomaly is :"
        )
        for timedelta in (
            predicted.index.max() -
            predicted_as_anomaly.dropna().index.sort_values(ascending=False)[:5]
        ):
            print(f"{timedelta} before")

    plt.title(title or "Predicted Result")
    plt.legend()
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(f'output/04/{title}.png')


def plot_predicted_results(
    result_df, original_df, sensor_id: Optional[str] = None,
    extra_result_df: Optional[pd.DataFrame] = None,
    threshold: Optional[float] = None,
    score_label: Optional[str] = None,
    extra_score_label: Optional[str] = None,
    broken_indices: Optional[pd.Index] = None,
    indices: Optional[pd.DatetimeIndex] = None,
    model: Optional[str] = None,
):
    # broken_indices = df[df['machine_status'] == 'BROKEN'].index
    # indices = (
    #     pd.DatetimeIndex([df.index[0]])
    #     .append(broken_indices)
    #     .append(pd.DatetimeIndex([df.index[-1]]))
    # )
    model = f"{model}:"
    for i in range(len(indices) - 1):
        sub_df = result_df.loc[indices[i]:indices[i+1]]
        if extra_result_df is not None:
            extra_sub_df = extra_result_df.loc[indices[i]:indices[i+1]]
        else:
            extra_sub_df = None

        if i == len(indices) - 1:
            title = "Predicted result to end"
        else:
            title = f"Predicted result to anomaly {i}"

        if len(sub_df.dropna()) > 0:

            plot_predicted_result(
                sub_df,
                original_df=original_df,
                sensor_id=sensor_id,
                extra_df=extra_sub_df,
                threshold=threshold,
                title=f"{model}Predicted result for anomaly {i}",
                score_label=score_label,
                extra_score_label=extra_score_label,
            )
