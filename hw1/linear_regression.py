import numpy as np
import matplotlib.pyplot as plt

def unit_matrix(n):
    unit = []
    for i in range(n):
        tmp = []
        for j in range(n):
            if (i==j):
                tmp.append(1)
            else:
                tmp.append(0)
        unit.append(tmp)
    return unit

def transpose(data):
    data_T = []
    for j in range(len(data[0])):
        tmp = []
        for i in range(len(data)):
            tmp.append(data[i][j])
        data_T.append(tmp)
    return (np.array(data_T))

def Gauss_inverse(data):
    X = np.copy(data)
    rows,cols=np.shape(data)
    I = unit_matrix(cols)
    for i in range(cols):
        scalar = 1.0 / X[i][i]
        for j in range(cols):
            X[i][j] *= scalar
            I[i][j] *= scalar
        for k in range(cols):
            if k != i:
                scalar_2 = X[k][i]
                for l in range(cols):
                     X[k][l] =X[k][l] -scalar_2*X[i][l]
                     I[k][l] =I[k][l] -scalar_2*I[i][l]
    return I

def design_matrix (x,n):
    design_m = np.zeros(shape=(len(x), n))
    for i,j in zip(range(n-1,-1,-1),range(n)):
        design_m[:,j] = np.array([x_data ** i for x_data in x]).flatten()
    return design_m

def predict_value(x,coe):
    predict = []
    for i in range(len(x)):
        tmp_pred = 0
        for j in range(len(coe)-1,-1,-1):
            tmp_pred += coe[len(coe)-1-j]*x[i]**j
        predict.append(tmp_pred)
    return predict

def calculate_err (x,coe,y):
    predict = predict_value(x,coe)
    error = 0
    for i in range(len(x)):
        error += (predict[i] - y[i])**2
    return error

def LSE(n,l,x,y):
    design_m = design_matrix(x,n)
    design_m_T = transpose(design_m)
    unit = unit_matrix(len(design_m[0]))
    ATA_lI = np.dot(design_m_T,design_m) + np.dot(l,unit)
    ATA_lI_inverse = Gauss_inverse(ATA_lI)
    LSE_coe = np.dot(np.dot(ATA_lI_inverse,design_m_T),y).flatten()
    error = calculate_err (x,LSE_coe,y)
    return LSE_coe,error

def fit_line(method,coe,error):
    result = f'{method}:\nFitting line: '
    for i in range(len(coe)):
        if (i == len(coe)-1):
            result = result + f'{coe[i]}'
        else:
            result = result + f'{coe[i]}X^{len(coe)-i-1} + '
    result = result +f'\nTotal error: {error[0]}'
    print(result)

def visualize(x,y,coe,method):
    fig = plt.figure ()
    plt.scatter(x, y, alpha=0.5,color = "red")
    predict = predict_value(x,coe)
    plt.plot(x,predict,color = "black")
    plt.savefig(f'./{method}_result.png')

def Newton(n,x,y):
    design_m = design_matrix(x,n)
    Newton_coe = np.random.rand(n,1)
    design_m2 = np.dot(transpose(design_m),design_m)
    hessian_m = design_m2*2
    while True:
        delta_x = np.dot(hessian_m,Newton_coe) - 2*np.dot(transpose(design_m),y)
        new_Newton_coe = Newton_coe - np.dot(Gauss_inverse(hessian_m),delta_x)
        error = calculate_err (x,Newton_coe,y)
        difference = sum([abs(new_Newton_coe[diff][0]-Newton_coe[diff][0]) for diff in range(n)])
        if difference <1e-5:
            break
        Newton_coe = new_Newton_coe
    Newton_coe = new_Newton_coe.flatten()
    return Newton_coe, error 

if __name__ == '__main__':
    data = np.hsplit(np.loadtxt('./testfile.txt', delimiter=','),[1,1])
    x = data[0]
    y = data[2]
    n = int(input("n :"))
    l = int(input("lambda :"))
    print()
    LSE_coe,LSE_err = LSE(n,l,x,y)
    fit_line("LSE",LSE_coe,LSE_err)
    visualize(x,y,LSE_coe,"LSE")
    print("\n")
    Newton_coe, Newton_err  = Newton(n,x,y)
    fit_line("Newton's Method",Newton_coe, Newton_err)
    visualize(x,y,Newton_coe,"Newton")



