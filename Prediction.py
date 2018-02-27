import LogisticRegression as log
import CsvUtils as csv
import numpy as np
from Preprocess import process_test_set


alpha = 0.0001  # learning rate
max_iterations = 300  # number of iteration


def main_process(train_path, test_path, selected_features):

    # 1）get training set data that used to train model
    train_data, train_label, m = log.get_train_data_and_label(train_path, selected_features)
    print("获取到的数据:", train_data, "----格式为：", train_data.shape)

    # 2）initialise the parameter of logistic regression model
    theta = log.initialise_logistic_regression_params(train_data)  # train_data is X
    print("初始化的权值格式", theta.shape)

    # 3)train the model to get the parameter theta (theta is a weight matrix)
    trained_theta = log.batch_gradient_descent(train_data, train_label, theta, alpha, max_iterations)
    print("训练后得到的权值为", trained_theta, "--格式为：", trained_theta.shape)

    # 4）get the test set data that used for prediction
    test_data, test_id = process_test_set(test_path, selected_features)
    print("测试数据为：", test_data, "---格式为：", test_data.shape)

    # 5）do prediction
    prediction_result_y = log.predict(test_data, trained_theta) # theta_default训练后的theta
    print("预测的结果为：", np.mat(prediction_result_y), "格式：", prediction_result_y.shape)

    # 6）transform prediction results into submission format
    result_list = []
    for i in range(len(prediction_result_y)):
        # print yPre[i, 0]
        if prediction_result_y[i, 0] >= 0.5:
            result_list.append(1)
        elif prediction_result_y[i, 0] < 0.5:
            result_list.append(0)
    y_array = np.array(result_list)

    return y_array, test_id


def start_prediction(train_path, test_path, prediction_path, selected_features):
    """
    It invokes main_process, forms the prediction result and stores the result to  prediction_path
    :param train_path: Path of training set
    :param test_path: Path of test set
    :param prediction_path: Path for storing prediction result
    :return:
    """
    prediction_result_y, id_in_test = main_process(train_path, test_path, selected_features)

    id_list = id_in_test[:, 0]
    id_and_its_prediction = zip(id_list, prediction_result_y)
    # print id_and_its_prediction
    columns_list = ["ID", "TARGET"]

    csv.write_csv(columns_list, list(id_and_its_prediction), prediction_path)  # end of the script


'''Start Here'''
train_path = "E:\\dataset\\train.csv"
test_path = "E:\\dataset\\test.csv"
prediction_path = "E:\\dataset\\sample_submission1.csv"  # the prediction result will be written to this path
selected_features = ["var38", 'var15', 'saldo_medio_var5_ult3', 'saldo_medio_var5_hace3', 'num_var45_ult3',
                     'saldo_medio_var5_hace2','saldo_var30', 'num_var45_hace3', 'num_var45_hace2', 'saldo_var42',
                     'num_var22_ult3','saldo_medio_var5_ult1','saldo_var5', 'num_var45_ult1', 'num_med_var45_ult3',
                     'num_var22_hace3', 'num_var22_hace2', 'var36','num_var22_ult1', 'num_meses_var39_vig_ult3']

start_prediction(train_path, test_path, prediction_path, selected_features)
