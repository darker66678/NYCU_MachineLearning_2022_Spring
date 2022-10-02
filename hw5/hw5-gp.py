import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import optimize as op

def get_input():
    f = open("./data/input.data")
    data = f.readlines()
    input_data = np.zeros((len(data), 2))
    for i in range(len(data)):
        tmp = data[i].split(' ')
        input_data[i, 0] = tmp[0]
        input_data[i, 1] = tmp[1]
    x = input_data[:, 0].reshape(-1, 1)
    y = input_data[:, 1].reshape(-1, 1)
    return x, y

def rational_quadratic(x1, x2, lengthscale=1.0, var=1.0, a=1.0):
    dis = distance.cdist(x1, x2, 'sqeuclidean')
    return var*(1+(dis/2*a*(lengthscale**2)))**-a


def gaussion_process(x, y, sample_x, beta, param=None):
    delta = np.identity(len(x))
    if(param is not None):
        C = rational_quadratic(x, x, param[0], param[1], param[2]) + ((beta**-1)*delta)
        mean = np.dot(rational_quadratic(x, sample_x, param[0], param[1], param[2]).T,
                    np.dot(np.linalg.inv(C), y))
        k_star = rational_quadratic(sample_x, sample_x, param[0], param[1], param[2])+(beta**-1)
        co_var = k_star - np.dot(rational_quadratic(x, sample_x, param[0], param[1], param[2]).T,
                                np.dot(np.linalg.inv(C), rational_quadratic(x, sample_x, param[0], param[1], param[2])))
    else:
        C = rational_quadratic(x, x) + ((beta**-1)*delta)
        mean = np.dot(rational_quadratic(x, sample_x).T,
                    np.dot(np.linalg.inv(C), y))
        k_star = rational_quadratic(sample_x, sample_x)+(beta**-1)
        co_var = k_star - np.dot(rational_quadratic(x, sample_x).T,
                                np.dot(np.linalg.inv(C), rational_quadratic(x, sample_x)))

    return mean, co_var


def plot(x, y, sample_x, mean, co_var,mode=None):
    x = x.reshape(-1)
    y = y.reshape(-1)
    sample_x = sample_x.reshape(-1)
    mean = mean.reshape(-1)
    std = co_var.diagonal()**0.5
    plt.figure()
    plt.scatter(x, y)
    plt.plot(sample_x, mean, c='r')
    up = mean+1.96*std
    low = mean-1.96*std
    plt.fill_between(sample_x, up, low, facecolor='purple', alpha=0.3)
    if(mode is not None):
        plt.savefig(f'./gaussion_process_{mode}.png')
    else:
        plt.savefig("./gaussion_process.png")



def optimize(x,y,beta):
    def negative_log_likelihood(param):
        C = rational_quadratic(x, x,param[0],param[1],param[2]) + ((beta**-1)*np.identity(len(x)))
        cost = 0.5*np.log(np.linalg.det(C)) + 0.5*np.dot(y.T,np.dot(np.linalg.inv(C), y)) + len(x)/2 * np.log(np.pi*2)
        return cost[0]
    param = np.ones(3)
    ans = op.minimize(negative_log_likelihood,param)
    print(ans)
    return ans.x

if __name__ == "__main__":
    # init
    beta = 5
    x, y = get_input()
    sample_x = np.linspace(-60, 60, 100).reshape(-1, 1)

    # task 1
    mean, co_var = gaussion_process(x, y, sample_x, beta)
    plot(x, y, sample_x, mean, co_var)

    # task 2
    best_param = optimize(x,y,beta)
    mean, co_var = gaussion_process(x, y, sample_x, beta, best_param)
    plot(x, y, sample_x, mean, co_var, "optimize")
    print("ok")
