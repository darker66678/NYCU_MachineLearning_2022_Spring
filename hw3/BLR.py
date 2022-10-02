import numpy as np
import random
import matplotlib.pyplot as plt
import logging


def univariate_gaussian(mean, var):
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    z = np.sqrt(-2*np.log(u)) * np.cos(2*np.pi*v)
    x = z*np.sqrt(var) + mean
    return x


def Polynomial_data(a, w_vector):
    e = univariate_gaussian(0, a)
    x = random.uniform(-1, 1)
    y = 0
    for i in range(len(w_vector)):
        y = y + (w_vector[i]*(x**i))
    y = y + e
    return x, y


def gen_x_matrix(x, n):
    m = []
    for i in range(n):
        m.append(x**i)
    return np.array(m).reshape(1, -1)


def ground_truth(w_vector):
    x = np.linspace(-2, 2, 100)
    y = []
    for i in x:
        val = 0
        for j in range(len(w_vector)):
            val = val + w_vector[j]*(i**j)
        y.append(val)
    return np.array(x), np.array(y)


def check_converge(post_mean, post_var, prior_mean, prior_var):
    change = np.sum(abs(prior_mean-post_mean)) + \
        np.sum(abs(post_var-prior_var))
    if change < 1e-5:
        return True
    else:
        return False


def predict_space(x, n, a, mean, var):
    predict_mean = []
    predict_upvar = []
    predict_lowvar = []
    for i in x:
        x_matrix = gen_x_matrix(i, n)
        d_m = round(np.dot(x_matrix, mean)[0][0], 10)
        d_v = round((a + np.dot(np.dot(x_matrix, var), x_matrix.T))[0][0], 10)
        predict_mean.append(d_m)
        predict_upvar.append(d_m + d_v)
        predict_lowvar.append(d_m-d_v)
    return predict_mean, predict_upvar, predict_lowvar


def plot(n, w_vector, a, data_x, data_y, mean_array, var_array):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Ground truth
    x, y = ground_truth(w_vector)
    axes[0, 0].set_xlim([-2, 2])
    axes[0, 0].set_ylim([-20, 20])
    axes[0, 0].set_title('Ground truth')
    axes[0, 0].plot(x, y, 'black')
    axes[0, 0].plot(x, y+a, 'red')
    axes[0, 0].plot(x, y-a, 'red')

    # all the data
    mean = mean_array[-1]
    var = var_array[-1]
    predict_mean, predict_upvar, predict_lowvar = predict_space(
        x, n, a, mean, var)
    axes[0, 1].set_xlim([-2, 2])
    axes[0, 1].set_ylim([-20, 20])
    axes[0, 1].set_title('Predict result')
    axes[0, 1].scatter(data_x, data_y, 1)
    axes[0, 1].plot(x, predict_mean, 'black')
    axes[0, 1].plot(x, predict_upvar, 'red')
    axes[0, 1].plot(x, predict_lowvar, 'red')

    # 10 data
    mean = mean_array[9]
    var = var_array[9]
    predict_mean, predict_upvar, predict_lowvar = predict_space(
        x, n, a, mean, var)
    axes[1, 0].set_xlim([-2, 2])
    axes[1, 0].set_ylim([-20, 20])
    axes[1, 0].set_title('After 10 incomes')
    axes[1, 0].scatter(data_x[:10], data_y[:10], 1)
    axes[1, 0].plot(x, predict_mean, 'black')
    axes[1, 0].plot(x, predict_upvar, 'red')
    axes[1, 0].plot(x, predict_lowvar, 'red')

    # 50 data
    mean = mean_array[49]
    var = var_array[49]
    predict_mean, predict_upvar, predict_lowvar = predict_space(
        x, n, a, mean, var)
    axes[1, 1].set_xlim([-2, 2])
    axes[1, 1].set_ylim([-20, 20])
    axes[1, 1].set_title('After 50 incomes')
    axes[1, 1].scatter(data_x[:50], data_y[:50], 1)
    axes[1, 1].plot(x, predict_mean, 'black')
    axes[1, 1].plot(x, predict_upvar, 'red')
    axes[1, 1].plot(x, predict_lowvar, 'red')

    plt.savefig('res.png')


def get_input():
    n = int(input("n = "))
    a = float(input("a = "))
    w_vector = []
    for i in range(n):
        w = float(input(f'w{i} = '))
        w_vector.append(w)
    b = float(input("b = "))
    return n, a, w_vector, b


def BLR():
    n, a, w_vector, b = get_input()
    # default param
    prior_mean = np.zeros((n, 1))
    prior_var = (1 / b) * np.identity(n)

    data_x = []
    data_y = []
    time = 0
    mean_array = []
    var_array = []

    while True:
        x, y = Polynomial_data(a, w_vector)
        logging.info(f'Add data point ({x}, {y}):\n')
        data_x.append(x)
        data_y.append(y)

        x_matrix = gen_x_matrix(x, n)
        post_var = np.linalg.inv(
            ((1/a)*np.dot(x_matrix.T, x_matrix))+np.linalg.inv(prior_var))
        post_mean = np.dot(post_var, (((1/a)*np.dot(x_matrix.T, y)) +
                           (np.dot(np.linalg.inv(prior_var), prior_mean))))

        logging.info("Postirior mean:")
        for i in post_mean:
            logging.info("   %.10f" % i[0])

        logging.info("\nPosterior variance:")
        for i in range(len(post_var)):
            output = ""
            for j in range(len(post_var[0])):
                output = output + "   " + str(round(post_var[i][j], 10))
            logging.info(output)
        if(time == 0):
            distribution_m = 0.0
        else:
            distribution_m = round(np.dot(x_matrix, post_mean)[0][0], 10)
        distribution_v = round(
            (a + np.dot(np.dot(x_matrix, post_var), x_matrix.T))[0][0], 10)

        logging.info(
            f'\nPredictive distribution ~ N({distribution_m}, {distribution_v})')
        logging.info('------------------------------------------------')

        mean_array.append(post_mean)
        var_array.append(post_var)

        time += 1
        if check_converge(post_mean, post_var, prior_mean, prior_var):
            break

        prior_mean = post_mean
        prior_var = post_var

    plot(n, w_vector, a, data_x, data_y, mean_array, var_array)


if __name__ == "__main__":
    fh = logging.FileHandler(encoding='utf-8', mode='w', filename='Bayes.log')
    logging.basicConfig(
        handlers=[fh], level=logging.DEBUG, format='%(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True
    BLR()
