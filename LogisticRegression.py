import numpy as np
from Preprocess import process_train_set


def get_train_data_and_label(train_path, selected_features):
    """
    It generates an matrix containing data used for training model
    :param train_path: Path of training set
    :param selected_features: Features that selected manually
    :return X: selected data set used to train model
             train_label: Feature 'TARGET' in training set
             m: number of samples
    """
    selected_data, train_label = process_train_set(train_path, selected_features)

    # get num of samples (m) and num of features (n)
    m, n = np.shape(selected_data)
    print("原始数据的格式：", selected_data.shape)

    # generate an m * (n+1) matrix whose elements are all 1; an extra column for x0
    x_train = np.ones((m, n + 1), dtype="complex")  # 这里似乎可以不是复数

    # put selected training set data into X
    x_train[:, :-1] = selected_data

    train_label = np.array(train_label, dtype="complex")  # 只是把label变为复数了

    return x_train, train_label, m


def initialise_logistic_regression_params(train_data):
    """
    Initialise parameter theta
    :param train_data: selected training set data
    :return theta: an initialised n*1 weight matrix whose elements are all 1
    """
    m, n = np.shape(train_data)
    theta = np.ones(n)  # theta is the weight matrix to be trained

    return theta


def batch_gradient_descent(train_data, train_label, theta, alpha, max_iterations):
    """
    This method implements the gradient descent algorithm to train the parameter theta
    :param train_data: selected training set data
    :param train_label: Labels provided in the training set
    :param theta: weights for the logistic regression model
    :param alpha: learning rate
    :param max_iterations: number of iterations
    :return trained_theta: weights after training the model
    """
    train_data_matrix = np.mat(train_data)  # m*(n+1)
    train_label_matrix_t = np.mat(train_label).T  # m*1
    trained_theta = np.mat(theta).T  # n*1

    for i in range(0, max_iterations):  # 迭代数据
        # calculate z = theta_0*x0 + theta_1*x1 + ... + theta_n*xn
        z = np.dot(train_data_matrix, trained_theta)
        # calculate h(x) = g(z) where g() is the sigmoid function
        sigmoid_matrix = np.mat(sigmoid(z))
        # calculate error/deviation
        error = train_label_matrix_t - sigmoid_matrix  # y - h(x)
        # update theta (theta_j = theta_j - alpha * x_T * error)
        trained_theta = trained_theta - alpha * (train_data_matrix.T * error)

    return trained_theta


def predict(test_data, theta):
    """
    This method implements the prediction
    :param test_data: new data used to do prediction
    :param theta: trained parameters
    :return prediction_result_y: the prediction result
    """
    m, n = np.shape(test_data)
    x_test = np.ones((m, n + 1), dtype="complex")
    x_test[:, :-1] = test_data

    # calculate z = theta_0*x0 + theta_1*x1 + ... + theta_n*xn
    prediction_result_y = np.dot(np.mat(x_test), theta)
    prediction_result_y = mean_normalization(prediction_result_y)
    # print("预测值归一化后：", prediction_result_y)
    # print("格式为：", prediction_result_y.shape)
    prediction_result_y = sigmoid(prediction_result_y)  # h(x) = g(z)

    return prediction_result_y


def mean_normalization(input_data):
    input_data = (input_data - np.mean(input_data)) / (np.max(input_data) - np.min(input_data))
    return input_data


def sigmoid(z):
    """
    This method implements a sigmoid function
    :param z: an intermediate result of the prediction
    :return: a value between 0 to 1
    """
    return 1.0 / (1 + np.exp(-z))

