import numpy as np
import random
import matplotlib.pyplot as plt


def univariate_gaussian(mean, var):
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    z = np.sqrt(-2*np.log(u)) * np.cos(2*np.pi*v)
    x = z*np.sqrt(var) + mean
    return x


def plt_result(x, y, gd_res, newton_res, n):
    fig, axes = plt.subplots(1, 3, figsize=(8, 8))

    color_pred = []
    for i in gd_res:
        if (i == 0):
            color_pred.append('r')
        else:
            color_pred.append('b')

    axes[0].set_title('Ground truth')
    axes[0].scatter(x[:n, 1], x[:n, 2], c='r')
    axes[0].scatter(x[n:, 1], x[n:, 2], c='b')

    axes[1].set_title('Gradient descent')
    axes[1].scatter(x[:, 1], x[:, 2], c=color_pred)

    newton_pred = []
    for i in newton_res:
        if (i == 0):
            newton_pred.append('r')
        else:
            newton_pred.append('b')

    axes[2].set_title('Newton\'s method')
    axes[2].scatter(x[:, 1], x[:, 2], c=newton_pred)
    plt.savefig('res.png')


def predict(x, w):
    res = np.dot(x, w)
    res[res >= 0.5] = 1
    res[res < 0.5] = 0
    return res


def CM(y, pred_res):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y)):
        if(y[i] == pred_res[i]):
            if(pred_res[i] == 1.0):
                TP += 1
            else:
                TN += 1
        else:
            if(pred_res[i] == 1.0):
                FP += 1
            else:
                FN += 1

    print("Confusion Matrix:")
    print('%50s' % "Predict cluster 1 Predict cluster 2")
    print('%-50s' % f'Is cluster 1           {TP}               {FN}')
    print('%-50s' % f'Is cluster 2           {FP}               {TN}')
    Sensitivity = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    print(f'\nSensitivity (Successfully predict cluster 1):{Sensitivity}')
    print(f'Specificity (Successfully predict cluster 2):{Specificity}\n')
    print("----------------------------------")


def sigmoid(val):
    val[val < -100] = -100
    res = 1/(1+np.exp(-val))
    return res


def gradient_descent(x, y, w, lr):
    time = 0
    while(True):
        time += 1
        update = np.dot(x.T, (sigmoid(np.dot(x, w))-y))
        w_next = w - (lr * update)
        if sum(abs(update)) < 1e-5:
            break
        if(time > 5000):
            break
        w = w_next
    pred_res = predict(x, w)
    print("\nGradient descent:\n")
    print(f'w: \n{w[0][0]}\n{w[1][0]}\n{w[2][0]}\n')
    CM(y, pred_res)
    return pred_res


def H_m_d(x, w):
    D = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        val = np.dot(x[i, :], w)
        if(val < -100):
            molecular = np.exp(-val)
            denominator = (1+molecular)**2
            value = molecular/denominator
            D[i, i] = value
    return D


def newton(x, y, w):
    time = 0
    while(True):
        time += 1
        D = H_m_d(x, w)
        h_m = np.dot(x.T, np.dot(D, x))
        delta_x = np.dot(x.T, (sigmoid(np.dot(x, w))-y))

        if np.linalg.det(h_m) == 0:
            update = 0.01 * delta_x
        else:
            update = np.dot(np.linalg.inv(h_m), delta_x)

        w_next = w - (update)

        if sum(abs(update)) < 1e-5:
            break
        if(time > 5000):
            break

        w = w_next

    pred_res = predict(x, w)
    print("\nNewton's method:\n")
    print(f'w: \n{w[0][0]}\n{w[1][0]}\n{w[2][0]}\n')
    CM(y, pred_res)
    return pred_res


def logistic():
    n = int(input("N = "))
    mx_1 = float(input("mx_1 = "))
    my_1 = float(input("my_1 = "))
    mx_2 = float(input("mx_2 = "))
    my_2 = float(input("my_2 = "))
    vx_1 = float(input("vx_1 = "))
    vy_1 = float(input("vy_1 = "))
    vx_2 = float(input("vx_2 = "))
    vy_2 = float(input("vy_2 = "))
    D1 = []
    D2 = []
    # prepare data
    for i in range(n):
        x1 = univariate_gaussian(mx_1, vx_1)
        y1 = univariate_gaussian(my_1, vy_1)
        x2 = univariate_gaussian(mx_2, vx_2)
        y2 = univariate_gaussian(my_2, vy_2)
        D1.append([1, x1, y1])
        D2.append([1, x2, y2])
    D1 = np.array(D1)
    D2 = np.array(D2)
    x = np.concatenate([D1, D2])
    y = np.zeros((2*n, 1))
    y[n:, 0] = 1
    w = np.array([[1], [1], [1]])
    lr = 0.01

    gd_res = gradient_descent(x, y, w, lr)
    newton_res = newton(x, y, w)
    plt_result(x, y, gd_res, newton_res, n)


if __name__ == "__main__":
    logistic()
