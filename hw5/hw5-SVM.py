import numpy as np
import matplotlib.pyplot as plt
import libsvm.svmutil as svmutil
import libsvm.svm as svm
import argparse
from scipy.spatial import distance

def args_init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=1, type=int)
    args = parser.parse_args()
    return args

def get_data():
    train_x = np.genfromtxt('./data/X_train.csv', delimiter=',', skip_header = 0)
    train_y = np.genfromtxt('./data/Y_train.csv', delimiter=',', skip_header = 0)
    test_x = np.genfromtxt('./data/X_test.csv', delimiter=',', skip_header = 0)
    test_y = np.genfromtxt('./data/Y_test.csv', delimiter=',', skip_header = 0)
    return train_x,train_y,test_x,test_y

def linear_rbf_kernel(train_x,test_x,gamma):
    new_train_x = np.zeros((5000,5001))
    new_test_x = np.zeros((2500,5001))
    # training
    linear_train_x = np.dot(train_x,train_x.T)
    rbf_train_x = np.exp((-gamma**2) * distance.cdist(train_x,train_x, 'sqeuclidean'))
    new_train_x[:,1:] = linear_train_x+rbf_train_x
    new_train_x[:,:1] = np.arange(5000)[:,np.newaxis]+1
    # testing
    linear_test_x = np.dot(test_x,train_x.T)
    rbf_test_x = np.exp((-gamma**2) * distance.cdist(test_x,train_x, 'sqeuclidean'))
    new_test_x[:,1:] = linear_test_x + rbf_test_x
    new_test_x[:,:1] = np.arange(2500)[:,np.newaxis]+1
    
    return new_train_x,new_test_x

def metrics(pred,true):
    TP=0
    FP=0
    TN=0
    FN=0
    for i in range(len(true)):
        if(true[i] == 1):
            if(true[i] == pred[i]):
                TP+=1
            else:
                FN+=1
        else:
            if(true[i] == pred[i]):
                TN+=1
            else:
                FP+=1
    acc = (TP+TN)/len(true)
    recall = TP/(TP+FN)
    precision = TP/(FP+TP)

    return acc,recall,precision

def SVM_1(train_x,train_y,test_x,test_y):
    kernel = ['linear','poly','rbf']
    for i in range(len(kernel)):
        print("-----------------------------")
        print(f'Kernel : {kernel[i]}')
        model = svmutil.svm_train(train_y,train_x,f'-q -t {i}')
        p_label, p_acc, p_val  = svmutil.svm_predict(test_y,test_x,model)
        acc,recall,precision = metrics(p_label,test_y)
        print(acc,recall,precision)


def grid_linear(k,train_y,train_x,test_y,test_x):
    rid_search_index = ['cost']
    grid_search = [[1,5,10,20,50,100]]
    for c in range(len(grid_search[0])):
        print(f'kernel: linear , cost: {grid_search[0][c]}')
        model = svmutil.svm_train(train_y,train_x,f'-q -s 0 -t {k} -c {grid_search[0][c]} -m 10000')
        p_label, p_acc, p_val  = svmutil.svm_predict(test_y,test_x,model)
        acc,recall,precision = metrics(p_label,test_y)
        print(acc,recall,precision)

def grid_poly(k,train_y,train_x,test_y,test_x):
    grid_search_index = ['cost','gamma','coef0','degree']
    grid_search = [[1,20,50],[0.001,0.01,0.1],[0,0.01,0.1],[1,3,10]]
    for c in range(len(grid_search[0])):
        for g in range(len(grid_search[1])):
            for coef in range(len(grid_search[2])):
                for degree in range(len(grid_search[3])):
                    print(f'kernel: poly , cost: {grid_search[0][c]} , gamma: {grid_search[1][g]} , coef0: {grid_search[2][coef]} , degree: {grid_search[3][degree]}')
                    model = svmutil.svm_train(train_y,train_x,f'-q -s 0 -t {k} -c {grid_search[0][c]} -g {grid_search[1][g]} -r {grid_search[2][coef]}  -m 10000 -d {grid_search[3][degree]}')
                    p_label, p_acc, p_val  = svmutil.svm_predict(test_y,test_x,model)
                    acc,recall,precision = metrics(p_label,test_y)
                    print(acc,recall,precision)

def grid_rbf(k,train_y,train_x,test_y,test_x):
    grid_search_index = ['cost','gamma']
    grid_search = [[1,20,50],[0.001,0.01,0.1]]
    for c in range(len(grid_search[0])):
        for g in range(len(grid_search[1])):
            print(f'kernel: rbf , cost: {grid_search[0][c]} , gamma: {grid_search[1][g]} ')
            model = svmutil.svm_train(train_y,train_x,f'-q -s 0 -t {k} -c {grid_search[0][c]} -g {grid_search[1][g]} -m 10000 ')
            p_label, p_acc, p_val  = svmutil.svm_predict(test_y,test_x,model)
            acc,recall,precision = metrics(p_label,test_y)
            print(acc,recall,precision)

'''def grid_sigmoid(k,train_y,train_x,test_y,test_x):
    grid_search_index = ['cost','gamma','coef0']
    grid_search = [[1,20,50],[0.001,0.01,0.1],[0,0.01,0.1]]
    for c in range(len(grid_search[0])):
        for g in range(len(grid_search[1])):
            for coef in range(len(grid_search[2])):
                print(f'kernel: sigmoid , cost: {grid_search[0][c]} , gamma: {grid_search[1][g]} , coef0: {grid_search[2][coef]}')
                model = svmutil.svm_train(train_y,train_x,f'-q -s 0 -t {k} -c {grid_search[0][c]} -g {grid_search[1][g]} -r {grid_search[2][coef]}  -m 10000')
                p_label, p_acc, p_val  = svmutil.svm_predict(test_y,test_x,model)
                acc,recall,precision = metrics(p_label,test_y)
                print(acc,recall,precision)'''

def SVM_2(train_x,train_y,test_x,test_y):
    kernel = ['linear','poly','rbf']
    for k in range(len(kernel)):
        if k == 0:
            grid_linear(k,train_y,train_x,test_y,test_x)
        elif k==1:
            grid_poly(k,train_y,train_x,test_y,test_x)
        elif k==2:
            grid_rbf(k,train_y,train_x,test_y,test_x)
        

def SVM_3(train_x,train_y,test_x,test_y):
    grid_search_index = ['cost','gamma']
    grid_search = [[1,20,50],[0.001,1/784,0.01,0.1]]
    for c in range(len(grid_search[0])):
        for g in range(len(grid_search[1])):
            print(f'cost: {grid_search[0][c]} , gamma: {grid_search[1][g]}')
            new_train_x,new_test_x = linear_rbf_kernel(train_x,test_x,grid_search[1][g])
            model = svmutil.svm_train(train_y,new_train_x,f'-q -s 0 -t 4 -c {grid_search[0][c]}  -m 10000')
            p_label, p_acc, p_val  = svmutil.svm_predict(test_y,new_test_x,model)
            acc,recall,precision = metrics(p_label,test_y)
            print(acc,recall,precision)
    
if __name__ == '__main__':
    args = args_init()
    train_x,train_y,test_x,test_y = get_data()
    if(args.mode == 1 ):
        SVM_1(train_x,train_y,test_x,test_y)
    if(args.mode == 2 ):
        SVM_2(train_x,train_y,test_x,test_y)
    if(args.mode == 3 ):
        SVM_3(train_x,train_y,test_x,test_y)