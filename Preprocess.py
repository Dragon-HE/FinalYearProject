import pandas as pd
import numpy as np
from Plot import plot_histogram


"""
数据需要做均值归一话，
（数值-均值）/（最大值-最小值）
"""


def read_csv(file_path):  # CsvUtils.py里也有一个read_csv()
    data = pd.read_csv(file_path)
    return data


def mean_normalization_formula(data_list, column_mean, column_max, column_min):  # 可以合并
    data_list = (data_list - column_mean) / (column_max - column_min)
    return data_list


def mean_normalization_for_data_set(data_frame):
    columns = data_frame.columns.values

    for index in range(len(columns)):
        # print column
        # print index
        column = columns[index]
        column_msg = data_frame[column]
        column_max = column_msg.max()
        column_min = column_msg.min()
        column_mean = column_msg.mean()
        # print column
        if column != '\'ID':
            if column_max > 10000:
                data_frame[column] = data_frame[column].apply(mean_normalization_formula,
                                                              args=(column_mean, column_max, column_min))

    return data_frame


def remove_constant_feature(data):
    """
    Remove constant features
    :param data: 待处理的数据集
    :return: 处理后的结果
    """
    removed = []
    for col in data.columns:
        if data[col].std() == 0:
            # std = 0 is a smart idea. # see https://www.kaggle.com/kobakhit/0-84-score-with-36-features-only/code
            removed.append(col)
    data.drop(removed, axis=1, inplace=True)  # drop columns and return new array
    print('\nAfter removing constant features: {}'.format(data.shape))
    return data


def remove_duplicate_feature(data):
    """
    Remove duplicate features
    :param data: 待处理的数据集
    :return: 处理后的结果
    """
    removed = []
    col = data.columns
    for i in range(len(col) - 1):
        val = data[col[i]].values
        for j in range(i + 1, len(col)):
            val2 = data[col[j]].values
            if np.array_equal(val, val2):
                removed.append(col[j])
    # print(removed)
    data.drop(removed, axis=1, inplace=True)
    print('\nAfter removing duplicate features: {}'.format(data.shape))
    return data


def rename_features(data):
    """
    Now rename the remaining features. use X_i to denote the ith feature
    :param data: 待处理的数据集
    :return: 处理后的结果
    """
    col = data.columns
    for i in range(len(col)):
        data.columns.values[i] = 'X_' + str(i + 1)
        # see https://stackoverflow.com/questions/43759921/pandas-rename-column-by-position
    print('\n''The names of the columns have been changed:')
    print(data.columns)
    return data


def remove_nan_sample(data):
    """
    All -999999 and 9999999999 have been replaced with nan. Delete samples that contain nan
    :param data:
    :return:
    """
    col = data.columns
    for i in range(len(col)):
        indexes = list(data[np.isnan(data[col[i]])].index)
        # isnan() returns a boolean array. # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.isnan.html
        data = data.drop(indexes)
    print('\nAfter removing nan samples: {}'.format(data.shape))
    return data


def process_train_set(train_path, selected_features):
    """
    Training set is processed in this method
    :param train_path: Path of training set
    :param selected_features: Features that selected manually
    :return selected_train_data: Used for model training;
             train_label: The feature 'TARGET' in training set
    """
    train_data = read_csv(train_path)

    train_data = remove_constant_feature(train_data)
    train_data = remove_duplicate_feature(train_data)
    train_data = remove_nan_sample(train_data)  # remove samples that contain nan

    # simple visualization
    plot_histogram(train_data)
    # plotRight(train_data, showColumn, title1)

    df = pd.DataFrame(train_data)
    df = mean_normalization_for_data_set(df)
    # print df
    selected_train_data = np.array(df[selected_features])
    train_label = np.array(df[list(df.columns.values)[-1]])

    return selected_train_data, train_label


def process_test_set(test_path, selected_features):
    """
    Test set is processed in this method
    :param test_path: Path of test set
    :param selected_features: Features that selected manually
    :return selected_test_data: Used for prediction;
             test_id: The feature 'ID' in test set
    """
    test_data = read_csv(test_path)

    test_data = remove_constant_feature(test_data)
    test_data = remove_duplicate_feature(test_data)

    df = pd.DataFrame(test_data)
    df = mean_normalization_for_data_set(df)
    selected_test_data = np.array(df[selected_features])
    test_id = np.array(df[list(df.columns.values)[0:1]])

    return selected_test_data, test_id
