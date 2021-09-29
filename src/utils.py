import pandas as pd
import numpy as np

def clean_names(df):
    """
        Rename the dataframe's column
    """
    df = df.reset_index()
    col_names = {
        "年月日": "date",
        "時刻": "time",
        "貯水量": "res_vol",
        "貯水率": "res_rate",
        "流域平均雨量": "avg_rainfall",
        "放流量": "outflow_vol",
        "流入量": "inflow_vol",
        "timestamp": "timestamp"
    }
    df = df.rename(columns=col_names)[[x for x in col_names.values()]]
    return df

def detect_missing_data(df):
    """
        Returns rows where data is missing
    """
    missing_data_indices = list(
        np.where((df["delta"] != 1) == True)[0])
    missing_data_indices
    missing_df = df[["timestamp", "date", "time", "res_vol",
                     "avg_rainfall", "outflow_vol",
                     "inflow_vol", "delta"]].iloc[missing_data_indices]
    return missing_df.iloc[1:]

def interpolate(df):
    """
        FIXME:
            Might be nice to inject some noise on interpolated data
    """
    if len(df) == 1:
        return df[["timestamp",
                    "res_vol", "avg_rainfall",
                    "inflow_vol", "outflow_vol"]]
    start, end, delta = df["timestamp"].min(), df["timestamp"].max(), 3600
    return df[["timestamp",
                "res_vol", "avg_rainfall",
                "inflow_vol", "outflow_vol",
               ]].set_index("timestamp").reindex(np.arange(start, end, delta)).interpolate()

def split(df, missing_df, T1):
    """
        - In case the data is contiguous, return the whole dataframes wrapped in a list
        - In case the data has gaps, split according to maximum threshold, and return 
            list of contiguous sequences.
    """
    split_idxs = missing_df[missing_df["delta"] > T1].index.tolist()
    if split_idxs:
        idxs = [0] + split_idxs + [len(df)]
        return [df.loc[idxs[i-1]: idxs[i]-1] for i in range(1, len(idxs))]
    else:
        return [df]

def interpolate_zeros(s):
    return s.replace(0, np.NaN).interpolate()

def preprocess_datetime(df):
    """
        - Must come after interpolate(), which would have already 
            corrected the missing timestamps. The latter will be held in the 
            dataframe's index
        
        - Extracts detailed datetime informatioon such as year, month, day, hour
        - Precomputes the one-hot version of the datetime data mentioned above
    """
    df["datetime"] = pd.to_datetime(df.index.astype(int), unit="s")

    df["year"], df["month"], df["day"], df["hour"] = \
        df["datetime"].dt.year, df["datetime"].dt.month, \
        df["datetime"].dt.day, df["datetime"].dt.hour
    
    # Precompute onehot representation of 'month', 'day' and hour
    month_dummies, day_dummies, hour_dummies = \
        pd.get_dummies(df["month"], prefix="month"), \
        pd.get_dummies(df["day"], prefix="day"), \
        pd.get_dummies(df["hour"], prefix="hour")
    df = pd.concat([df, month_dummies, day_dummies, hour_dummies], axis=1)

    return df

def preprocess_func(raw_df, max_missing=10, min_data_seq_len=2160 + 72 + 1):
    """
        FIXME:
            Return easy to parse metadata on filtered sequences.
            This will help us troubleshoot bad data.
        - min_data_seq_len: minimum length of a contiguous sequence to be used
            by the agent
    """
    raw_df["delta"] = (raw_df.timestamp.diff() // 3600).fillna(1)
    raw_df["res_vol"] = interpolate_zeros(raw_df["res_vol"])
    missing_df = detect_missing_data(raw_df)
    seq_df = split(raw_df, missing_df, max_missing)
    seq_df = [interpolate(df) for df in seq_df]
    seq_df = [x for x in seq_df if len(x) >= min_data_seq_len]
    seq_df = [preprocess_datetime(x) for x in seq_df]
    return seq_df

def infer_max_volume(df):
    # TODO: Consider the need for denoising on "貯水量" and "貯水率"
    return (df["res_vol"]/df["res_rate"]).median() * 100