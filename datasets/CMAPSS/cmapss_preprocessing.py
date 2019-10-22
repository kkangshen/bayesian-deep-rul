# -*- coding: utf-8 -*-
"""C-MAPSS preprocessing."""

import os
import zipfile

import numpy as np
np.random.seed(seed=42)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib


def build_train_data(df, out_path, window=30, normalization="min-max"):
    """Build train data.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    out_path : str
        Output path.
    window : int, optional
        Sliding window size.
    normalization : str, optional
        Normalization strategy. Either 'min-max' or 'z-score'.

    Returns
    -------
    MinMaxScaler or StandardScaler
        Scaler used to normalize the data.
    """
    assert normalization in ["z-score", "min-max"], "'normalization' must be either 'z-score' or 'min-max', got '" + normalization + "'."

    # normalize data
    if normalization == "z-score":
        scaler = StandardScaler()
        df.loc[:, 1 : df.shape[1]] = scaler.fit_transform(df.iloc[:, 1 : df.shape[1]])
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df.loc[:, 1 : df.shape[1]] = scaler.fit_transform(df.iloc[:, 1 : df.shape[1]])

    # group by trajectory
    grouped = df.groupby("trajectory_id")

    # compute trajectory min and max length
    min_length = min(len(traj) for traj_id, traj in grouped)
    max_length = max(len(traj) for traj_id, traj in grouped)

    # count total number of samples
    total_samples = 0
    for traj_id, traj in grouped:
        t = traj.drop(["trajectory_id"], axis=1).values
        total_samples += len(t) - window + 1

    # print info
    print("Number of trajectories = " + str(len(grouped)))
    print("Trajectory min length = " + str(min_length))
    print("Trajectory max length = " + str(max_length))
    print("Number of features = " + str(len(df.columns) - 1))
    print("Number of samples = " + str(total_samples))

    assert window <= min_length, "'window' cannot be greater than " + str(min(len(traj) for traj_id, traj in grouped)) + ", got %d." % window

    # sample each trajectory through sliding window segmentation
    sample_id = 0
    for traj_id, traj in grouped:
        t = traj.drop(["trajectory_id"], axis=1).values
        num_samples = len(t) - window + 1
        for i in range(num_samples):
            sample = t[i : (i + window)]
            label = len(t) - i - window
            path = os.path.join(out_path, "train")
            if not os.path.exists(path): os.makedirs(path)
            file_name = os.path.join(path, "train_{0:0=3d}-{1:0=3d}.txt".format(sample_id, label))
            sample_id += 1
            np.savetxt(file_name, sample, fmt="%.10f")
    
    print("Done.")
    return scaler


def build_validation_data(df, out_path, scaler, window=30):
    """Build validation data.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    out_path : str
        Output path.
    scaler : MinMaxScaler or StandardScaler
        Scaler to use to normalize the data.
    window : int, optional
        Sliding window size.
    """
    assert scaler, "'scaler' type cannot be None."

    # normalize data    
    df.loc[:, 1 : df.shape[1]] = scaler.transform(df.iloc[:, 1 : df.shape[1]])

    # group by trajectory
    grouped = df.groupby("trajectory_id")

    # compute trajectory min and max length
    min_length = min(len(traj) for traj_id, traj in grouped)
    max_length = max(len(traj) for traj_id, traj in grouped)

    # count total number of samples
    total_samples = 0
    for traj_id, traj in grouped:
        t = traj.drop(["trajectory_id"], axis=1).values
        total_samples += len(t) - window + 1

    # print info
    print("Number of trajectories = " + str(len(grouped)))
    print("Trajectory min length = " + str(min_length))
    print("Trajectory max length = " + str(max_length))
    print("Number of features = " + str(len(df.columns) - 1))
    print("Number of samples = " + str(total_samples))

    assert window <= min_length, "'window' cannot be greater than " + str(min(len(traj) for traj_id, traj in grouped)) + ", got %d." % window

    # sample each trajectory through sliding window segmentation
    sample_id = 0
    for traj_id, traj in grouped:
        t = traj.drop(["trajectory_id"], axis=1).values
        num_samples = len(t) - window + 1
        for i in range(num_samples):
            sample = t[i : (i + window)]
            label = len(t) - i - window          
            path = os.path.join(out_path, "validation")
            if not os.path.exists(path): os.makedirs(path)
            file_name = os.path.join(path, "validation_{0:0=3d}-{1:0=3d}.txt".format(sample_id, label))
            sample_id += 1
            np.savetxt(file_name, sample, fmt="%.10f")

    print("Done.")


def build_test_data(df, file_rul, out_path, scaler, window=30, keep_all=False):
    """Build test data.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    file_rul : str
        RUL labels file.
    out_path : str
        Output path.
    scaler : MinMaxScaler or StandardScaler
        Scaler to use to normalize the data.
    window : int, optional
        Sliding window size.
    keep_all : bool, optional
        True to keep all the segments extracted from the series, False to keep only the last one.
    """
    assert scaler, "'scaler' type cannot be None."

    # normalize data    
    df.loc[:, 1 : df.shape[1]] = scaler.transform(df.iloc[:, 1 : df.shape[1]])

    # group by trajectory
    grouped = df.groupby("trajectory_id")

    # compute trajectory min and max length
    min_length = min(len(traj) for traj_id, traj in grouped)
    max_length = max(len(traj) for traj_id, traj in grouped)

    # count total number of samples
    total_samples = 0
    if keep_all:
        for traj_id, traj in grouped:
            t = traj.drop(["trajectory_id"], axis=1).values
            total_samples += len(t) - window + 1
    else:
        total_samples = len(grouped)

    # print info
    print("Number of trajectories = " + str(len(grouped)))
    print("Trajectory min length = " + str(min_length))
    print("Trajectory max length = " + str(max_length))
    print("Number of features = " + str(len(df.columns) - 1))
    print("Number of samples = " + str(total_samples))

    assert window <= min_length, "'window' cannot be greater than " + str(min(len(traj) for traj_id, traj in grouped)) + ", got %d." % window
    
    # get ground truth
    rul = np.asarray(file_rul.readlines(), dtype=np.int32)   

    # sample each trajectory through sliding window segmentation
    sample_id = 0
    if not keep_all:
        for traj_id, traj in grouped:
            t = traj.drop(["trajectory_id"], axis=1).values
            sample = t[-window :]
            label = rul[traj_id - 1]
            path = os.path.join(out_path, "test")
            if not os.path.exists(path): os.makedirs(path)
            file_name = os.path.join(path, "test_{0:0=3d}-{1:0=3d}.txt".format(sample_id, label))
            sample_id += 1
            np.savetxt(file_name, sample, fmt="%.10f")
    else:
        for traj_id, traj in grouped:
            t = traj.drop(["trajectory_id"], axis=1).values
            num_samples = len(t) - window + 1
            for i in range(num_samples):
                sample = t[i : (i + window)]
                label = len(t) - i - window + rul[traj_id - 1]
                path = os.path.join(out_path, "test")   
                if not os.path.exists(path): os.makedirs(path)
                file_name = os.path.join(path, "test_{0:0=3d}-{1:0=3d}.txt".format(sample_id, label))
                sample_id += 1
                np.savetxt(file_name, sample, fmt="%.10f")
    
    print("Done.")


def extract_dataframes(file_train, file_test, subset="FD001", validation=0.00):
    """Extract train, validation and test dataframe from source file.
    
    Parameters
    ----------
    file_train : str
        Training samples file.
    file_test : str
        Test samples file.
    subset: str, optional
        Subset. Either 'FD001' or 'FD002' or 'FD003' or 'FD004'.
    validation : float, optional
        Ratio of training samples to hold out for validation.
    
    Returns
    -------
    (DataFrame, DataFrame, DataFrame)
        Train dataframe, validation dataframe, test dataframe.
    """
    assert subset in ["FD001", "FD002", "FD003", "FD004"], "'subset' must be either 'FD001' or 'FD002' or 'FD003' or 'FD004', got '" + subset + "'."

    assert 0 <= validation <= 1, "'validation' must be a value within [0, 1], got %.2f" % validation + "."
  
    df = _load_data_from_file(file_train, subset=subset)
    
    # group by trajectory
    grouped = df.groupby("trajectory_id")

    df_train = []
    df_validation = []
    for traj_id, traj in grouped:
        # randomize train/validation splitting
        if np.random.rand() <= (validation + 0.1) and len(df_validation) < round(len(grouped) * validation):
            df_validation.append(traj)
        else:
            df_train.append(traj)

    # print info
    print("Number of training trajectories = " + str(len(df_train)))
    print("Number of validation trajectories = " + str(len(df_validation)))

    df_train = pd.concat(df_train)

    if len(df_validation) > 0:
        df_validation = pd.concat(df_validation)

    df_test = _load_data_from_file(file_test, subset=subset)

    print("Done.")
    return df_train, df_validation, df_test

    
def _load_data_from_file(file, subset="FD001"):
    """Load data from source file into a dataframe.
    
    Parameters
    ----------
    file : str
        Source file.
    subset: str, optional
        Subset. Either 'FD001' or 'FD002' or 'FD003' or 'FD004'.
    
    Returns
    -------
    DataFrame
        Data organized into a dataframe.
    """
    assert subset in ["FD001", "FD002", "FD003", "FD004"], "'subset' must be either 'FD001' or 'FD002' or 'FD003' or 'FD004', got '" + subset + "'."

    n_operational_settings = 3
    n_sensors = 21

    # read csv
    df = pd.read_csv(file, sep=" ", header=None, index_col=False).fillna(method="bfill")
    df = df.dropna(axis="columns", how="all")

    assert df.shape[1] == n_operational_settings + n_sensors + 2, "Expected %d columns, got %d." % (n_operational_settings + n_sensors + 2, df.shape[1])
    
    df.columns = ["trajectory_id", "t"] + ["setting_" + str(i + 1) for i in range(n_operational_settings)] + ["sensor_" + str(i + 1) for i in range(n_sensors)]

    # drop t
    df = df.drop(["t"], axis=1)

    if subset in ["FD001", "FD003"]:
        # drop operating_modes
        df = df.drop(["setting_" + str(i + 1) for i in range(n_operational_settings)], axis=1)

        # drop sensors which are useless according to the literature
        to_drop = [1, 5, 6, 10, 16, 18, 19]
        df = df.drop(["sensor_" + str(d) for d in to_drop], axis=1)

    return df


if __name__ == "__main__":
    """Preprocessing."""
    normalization = "min-max"
    validation = 0.00

    for subset, window in [("FD001", 30), ("FD002", 20), ("FD003", 30), ("FD004", 15)]:
        print("**** %s ****" % subset)
        print("normalization = " + normalization)
        print("window = " + str(window))
        print("validation = " + str(validation))

        # read .zip file into memory
        with zipfile.ZipFile("CMAPSSData.zip") as zip_file:
            file_train = zip_file.open("train_" + subset + ".txt")
            file_test = zip_file.open("test_" + subset + ".txt")
            file_rul = zip_file.open("RUL_" + subset + ".txt")

        print("Extracting dataframes...")
        df_train, df_validation, df_test = extract_dataframes(file_train=file_train, file_test=file_test, subset=subset, validation=validation)

        # build train data
        print("Preprocessing training data...")
        scaler = build_train_data(df=df_train, out_path="data/" + subset + "/" + normalization, window=window, normalization=normalization)

        # build validation data
        if len(df_validation) > 0:
            print("Preprocessing validation data...")
            build_validation_data(df=df_validation, out_path="data/" + subset + "/" + normalization, scaler=scaler, window=window)

        # build test data
        print("Preprocessing test data...")
        build_test_data(df=df_test, file_rul=file_rul, out_path="data/" + subset + "/" + normalization, scaler=scaler, window=window, keep_all=False)

        # save scaler
        print("Saving scaler object to file...")
        scaler_filename = "data/" + "/" + subset + "/" + normalization + "/scaler.sav"
        joblib.dump(scaler, scaler_filename)

        # close files
        file_train.close()
        file_test.close()
        file_rul.close()
        print("Done.")
