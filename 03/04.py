import pandas as pd
from function_sensor_data import (
    check_machine_status_transactions,
    plot_anomaly_list,
    add_stabilization_status,
    generate_lag_features,
    genereate_moving_average_features,
    generate_diff_features,
    select_features
)

# Load the dataset
df = pd.read_csv('data/sensor.csv')
print(df.head())


# timestamp column
df['timestamp'] = df['timestamp'].astype('datetime64[ns]')
df = df.set_index('timestamp')

print(df.index.min())
print(df.index.max())
print(df.index.max() - df.index.min())
print(len(df.index.drop_duplicates()))

# machine_status column
print(df['machine_status'].value_counts(dropna=False))

# Check the machine status transitions
check_machine_status_transactions(df)

print(df.describe().loc['count'].sort_values()[:3])

plot_anomaly_list(df, s_list=["s00", "s01"])

# 細かい粒度でのプロット
broken_indices = df[df['machine_status'] == 'broken'].index
indices = (
    pd.DatetimeIndex([df.index[0]])
    .append(broken_indices)
    .append(pd.DatetimeIndex([df.index[-1]]))
)

for i in range(len(indices) -1):
    plot_anomaly_list(
        df.loc[indices[i]:indices[i+1]],
        s_list=["s00"],
        title=f"anomaly {i}"
    )

df = add_stabilization_status(df)

df = (
    pd.read_csv('data/sensor.csv',
    parse_dates=['timestamp'],
    index_col='timestamp'
    )
    .bfill()
    .ffill()
    .pipe(add_stabilization_status)
)

tmp_df = df[["s00"]]
print(pd.concat([
    tmp_df,
    generate_lag_features(tmp_df[["s00"]], periods=1)
], axis=1).head())


# broken_indices がラベル(DatetimeIndex 等)の想定
if len(broken_indices) < 2:
    print("broken 区間が見つかりませんでした（len < 2）")
else:
    start_label, end_label = broken_indices[0], broken_indices[1]
    tmp_df = df.loc[start_label:end_label, ["s00", "machine_status"]]
    plot_anomaly_list(
        pd.concat([tmp_df, generate_diff_features(tmp_df[["s00"]], periods=1)], axis=1)
    )

tmp_df = df.loc[:broken_indices[0], ["s06", "machine_status"]]
plot_anomaly_list(
    pd.concat([
        tmp_df,
        generate_lag_features(tmp_df[["s06"]], periods=10),
        generate_lag_features(tmp_df[["s06"]], periods=10).pipe(
            genereate_moving_average_features,
            periods=10
        )
    ],axis=1)
)

tmp_df = df.loc[:broken_indices[0], ["s06", "machine_status"]]
plot_anomaly_list(
    pd.concat([
        tmp_df,
        generate_diff_features(tmp_df[["s06"]], periods=10),
        generate_diff_features(tmp_df[["s06"]], periods=10).pipe(
            genereate_moving_average_features,
            periods=10
        )
    ],axis=1)
)

normal_df = df[df['machine_status'] == 'NORMAL'].drop(columns=['machine_status'])
feature_df, selected_columns = select_features(normal_df)
