import pandas as pd
import numpy as np


def read_csv(file_path):
    """
    读取csv文件，不读取第一个列（序号列表）
    :param file_path: csv文件路径
    :return: 除第一列的数据
    """
    data = pd.read_csv(file_path)
    csv_data = pd.DataFrame(data)
    csv_data = csv_data.dropna()
    feature_id = np.array(csv_data[list(csv_data.columns.values)[0:1]])
    data_without_id = np.array(csv_data[list(csv_data.columns.values)[1:]])
    return data_without_id, feature_id


def write_csv(columns_list, data_list, file_path):
    """
    将数据写入csv文件中
    :param columns_list: 写入的列表名称
    :param data_list: 写入的数据
    :param file_path: 写入的文件地址
    :return:
    """
    f = open(file_path, 'w')
    dataframe = pd.DataFrame(columns=columns_list, data=data_list)
    dataframe.to_csv(file_path, index_label=False, index=False)
    f.close()
