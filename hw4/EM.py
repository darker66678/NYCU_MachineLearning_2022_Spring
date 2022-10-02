import numpy as np
import struct
from tqdm import tqdm, trange
import logging


def read_file(file, type):
    if type == "idx3":
        data = open(file, 'rb').read()
        offset = 0
        fmt_header = '>iiii'
        _, num_imgs, row, col = struct.unpack_from(fmt_header, data, offset)
        img_size = row * col
        image_fmt = ">"+str(img_size)+'B'
        offset = offset + struct.calcsize(fmt_header)
        all_imgs = np.empty((num_imgs, row, col))
        for i in range(num_imgs):
            img_data = struct.unpack_from(image_fmt, data, offset)
            all_imgs[i] = np.array(img_data).reshape(row, col)
            offset = offset + struct.calcsize(image_fmt)
        return all_imgs
    elif type == "idx1":
        data = open(file, 'rb').read()
        offset = 0
        fmt_header = '>ii'
        label_fmt = '>B'
        _, num_imgs = struct.unpack_from(fmt_header, data, offset)
        offset = offset + struct.calcsize(fmt_header)
        labels = np.empty(num_imgs)
        for i in range(num_imgs):
            iabel_data = struct.unpack_from(label_fmt, data, offset)
            labels[i] = iabel_data[0]
            offset = offset + struct.calcsize(label_fmt)
        return labels


def load_data():
    file = {"train_x": "train-images-idx3-ubyte",
            "train_y": "train-labels-idx1-ubyte",
            "test_x": "t10k-images-idx3-ubyte",
            "test_y": "t10k-labels-idx1-ubyte"}
    train_x = read_file(file["train_x"], "idx3")
    train_y = read_file(file["train_y"], "idx1")
    return train_x, train_y


def CM(y, pred_res):
    for j in range(10):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(y)):
            if(y[i] == j):
                if(pred_res[i] == j):
                    TP += 1
                else:
                    FN += 1
            else:
                if(pred_res[i] == j):
                    FP += 1
                else:
                    TN += 1

        logging.info(f'Confusion Matrix: {j}:')
        logging.info('%50s' % f'Predict number {j} Predict not number {j}')
        logging.info('%-50s' %
                     f'Is number {j}           {TP}               {FN}')
        logging.info('%-50s' %
                     f'Isn\'t number {j}           {FP}               {TN}')
        Sensitivity = TP/(TP+FN)
        Specificity = TN/(TN+FP)
        logging.info(
            f'\nSensitivity (Successfully predict number {j}):{Sensitivity}')
        logging.info(
            f'Specificity (Successfully predict not number {j}):{Specificity}\n')
        logging.info("----------------------------------\n")


def init_param(train_x):
    P = np.random.rand(10, train_x.shape[1])*0.4 + 0.2  # 10*784
    k = np.ones(10)*0.1  # 10*784
    return P, k


def E_step(train_x, P, k):
    W = np.zeros((train_x.shape[0], 10))  # 60000*10
    for data_num in range(train_x.shape[0]):  # 60000
        for number in range(len(k)):  # 10
            W[data_num, number] = np.prod(
                (P[number] ** train_x[data_num]) * ((1-P[number]) ** (1-train_x[data_num]))) * k[number]

    sum_w_by_data = W.sum(axis=1).reshape(-1, 1)
    sum_w_by_data[sum_w_by_data == 0] = 0.8
    W /= sum_w_by_data
    return W


def M_step(train_x, W):
    P = np.zeros((10, train_x.shape[1]))  # 10*784
    sum_w_by_label = W.sum(axis=0).reshape(-1, 1)
    sum_w_by_label[sum_w_by_label == 0] = 0.8
    k = W.sum(axis=0)/W.shape[0]
    for number in range(len(k)):
        for pixel in range(P.shape[1]):
            P[number, pixel] = np.dot(
                W[:, number].T, train_x[:, pixel]) / sum_w_by_label[number]
    return k, P


def plot_number(P, true_num=None):
    for i in range(P.shape[0]):
        if(true_num is None):
            logging.info(f'class: {i}:')
            location = i
        else:
            logging.info(f'labeled class {i}:')
            location = int(true_num[i])
        row = ''
        P_threshold = np.percentile(P[location, :], 90)
        for j in range(P.shape[1]):
            if(P[location, j] >= P_threshold):
                row += str(1)
            else:
                row += str(0)
            row += ' '
            if((j+1) % 28 == 0):
                logging.info(row)
                row = ''
        logging.info(" ")


def clean(res, x, y, i):
    res[x, :] = -100*(i+1)
    res[:, y] = -100*(i+1)
    return res


def find_true_number(train_x, train_y, P, k):
    W = np.zeros((train_x.shape[0], 10))  # 60000*10
    for data_num in range(train_x.shape[0]):  # 60000
        for number in range(len(k)):  # 10
            W[data_num, number] = np.prod(
                P[number] ** train_x[data_num] * (1-P[number]) ** (1-train_x[data_num])) * k[number]

    pred = np.argmax(W, axis=1)
    res = np.zeros((10, 10))
    true_num = np.zeros(10)
    for i in range(train_x.shape[0]):
        res[int(train_y[i]), pred[i]] += 1
    for i in range(10):
        x, y = np.where(res == res.max())
        true_num[x[0]] = y[0]
        res = clean(res, x[0], y[0], i)
    return true_num, W


def predict(W, true_num):
    pred = np.argmax(W, axis=1)
    pred_res = np.zeros(W.shape[0])
    for i in range(W.shape[0]):
        pred_res[i] = np.where(true_num == pred[i])[0][0]
    return pred_res


def get_error_rate(train_y, y_pred):
    error = 0
    for i in range(len(train_y)):
        if(train_y[i] != y_pred[i]):
            error += 1
    error_rate = error/len(train_y)
    return error_rate


def EM(train_x, train_y):
    P, k = init_param(train_x)  # init
    difference = 0
    for i in tqdm(range(50)):
        W = E_step(train_x, P, k)  # E step
        new_k, new_P = M_step(train_x, W)  # M step
        new_difference = np.abs(P-new_P).sum()
        plot_number(new_P)
        logging.info(
            f'No. of Iteration: {i+1}, Difference: {new_difference}\n\n\n')
        P = new_P
        k = new_k
        iteration = i+1
        if np.abs(difference - new_difference) < 0.5:
            break
        difference = new_difference

    logging.info("----------------------------------------")
    logging.info("----------------------------------------\n")
    true_num, likelihood = find_true_number(train_x, train_y, P, k)
    plot_number(P, true_num)
    y_pred = predict(likelihood, true_num)
    CM(train_y, y_pred)
    error_rate = get_error_rate(train_y, y_pred)
    logging.info(f'Total iteration to converge: {iteration}')
    logging.info(f'Total error rate: {error_rate}')


if __name__ == "__main__":
    fh = logging.FileHandler(encoding='utf-8', mode='w', filename='EM.log')
    logging.basicConfig(
        handlers=[fh], level=logging.DEBUG, format='%(message)s')
    train_x, train_y = load_data()
    train_x = train_x.reshape(
        (train_x.shape[0], train_x.shape[1]*train_x.shape[2]))
    train_x = train_x // 128  # 2 bin
    EM(train_x, train_y)
