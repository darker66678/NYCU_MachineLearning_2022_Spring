import numpy as np
import scipy.spatial.distance as distance
import cv2
import os
import glob
import matplotlib.pyplot as plt
import argparse
import itertools
from PIL import Image

def load_data():
    Training_path = "./Yale_Face_Database/Training/"
    Testing_path = "./Yale_Face_Database/Testing/"
    
    test_file = glob.glob(Testing_path+"*.pgm")
    train_file = glob.glob(Training_path+"*.pgm")
    tmp = cv2.imread(train_file[0],0);
    training_data = np.zeros((len(train_file),RESIZE[1],RESIZE[0]))
    testing_data = np.zeros((len(test_file),RESIZE[1],RESIZE[0]))
    training_label = np.zeros(len(train_file))
    testing_label = np.zeros(len(test_file))

    for f in range(len(train_file)):
        tmp = cv2.imread(train_file[f],0);
        tmp = cv2.resize(tmp, (RESIZE[0], RESIZE[1]), interpolation=cv2.INTER_AREA)
        training_data[f] = tmp
        training_label[f] = int(train_file[f].split("/")[-1].split(".")[0][7:9])
    for f in range(len(test_file)):
        tmp = cv2.imread(test_file[f],0);
        tmp = cv2.resize(tmp, (RESIZE[0],RESIZE[1]), interpolation=cv2.INTER_AREA)
        testing_data[f] = tmp
        testing_label[f] = int(test_file[f].split("/")[-1].split(".")[0][7:9])

    training_data = training_data.reshape(training_data.shape[0],training_data.shape[1]*training_data.shape[2])
    testing_data = testing_data.reshape(testing_data.shape[0],testing_data.shape[1]*testing_data.shape[2])

    return training_data,testing_data,training_label,testing_label

def show_eig_face(eig_vectors_n, algo):
    row = int(np.sqrt(N_component))
    fig, ax = plt.subplots(row,row)
    for i in range(eig_vectors_n.shape[1]):
        img = eig_vectors_n[:,i].reshape(RESIZE[0],RESIZE[1])
        ax[i//row][i%row].imshow(img, cmap='gray')
        ax[i//row][i%row].axis('off')

    plt.savefig(f'./eig_faces/{algo}.png')

def reconstruct(data,data_scaled,eig_vectors_n, algo, rand_idx ,mean, train_label=0):
    random_data = data_scaled[rand_idx,:]
    res_images = np.dot(np.dot(random_data,eig_vectors_n),eig_vectors_n.T) + mean
    
    fig, ax = plt.subplots(2,5)
    for i in range(rand_idx.shape[0]):
        img = res_images[i,:].reshape(RESIZE[0],RESIZE[1])
        if(i<5):
            ax[0][i%5].imshow(img, cmap='gray')
            ax[0][i%5].axis('off')
        else:
            ax[1][i%5].imshow(img, cmap='gray')
            ax[1][i%5].axis('off')

    plt.savefig(f'./random_pick/{algo}.png')

    # original data
    random_data_ori = data[rand_idx,:]
    fig, ax = plt.subplots(2,5)
    for i in range(rand_idx.shape[0]):
        img = random_data_ori[i,:].reshape(RESIZE[0],RESIZE[1])
        if(i<5):
            ax[0][i%5].imshow(img, cmap='gray', vmin = 0, vmax = 255,interpolation='none')
            ax[0][i%5].axis('off')
        else:
            ax[1][i%5].imshow(img, cmap='gray', vmin = 0, vmax = 255,interpolation='none')
            ax[1][i%5].axis('off')

    plt.savefig(f'./random_pick/original.png')

def PCA(train_data, rand_idx):
    # calculate mean_face
    mean = train_data.mean(axis = 0)
    train_data_scaled = train_data - mean
    # covariance matrix
    cov_matrix = np.dot(train_data_scaled.T,train_data_scaled)
    # solve eigen_problem
    eig_values, eig_vectors = np.linalg.eigh(cov_matrix)
    # pick 25 eigen_vectors
    idx = eig_values.argsort()[::-1][:N_component]
    eig_vectors_n = eig_vectors[:,idx].astype(np.float64)

    show_eig_face(eig_vectors_n,"PCA")
    reconstruct(train_data,train_data_scaled,eig_vectors_n, "PCA", rand_idx ,mean)
    
    return eig_vectors_n

def LDA(train_data, train_label, rand_idx):
    #S_w and S_b
    label_mean =np.zeros((15,RESIZE[0]*RESIZE[1]))
    S_w = np.zeros((RESIZE[0]*RESIZE[1],RESIZE[0]*RESIZE[1]))
    S_b = np.zeros((RESIZE[0]*RESIZE[1],RESIZE[0]*RESIZE[1]))
    all_mean = np.mean(train_data,axis=0)

    for i in range(15):
        idx = np.where(train_label == i+1)[0]
        label_mean[i,:] = np.mean(train_data[idx,:],axis=0)
        wi = (train_data[idx,:]-label_mean[i,:])
        S_w += np.dot(wi.T,wi)

        bi = (label_mean[i,:] - all_mean).reshape(-1,1)
        S_b += len(idx) * np.dot(bi,bi.T)
        
    matrix = np.dot(np.linalg.pinv(S_w),S_b)

    eig_values, eig_vectors = np.linalg.eigh(matrix)
    idx = eig_values.argsort()[::-1][:N_component]
    eig_vectors_n = eig_vectors[:,idx].astype(np.float64)

    show_eig_face(eig_vectors_n,"LDA")
    train_data_scaled = train_data-all_mean
    reconstruct(train_data,train_data_scaled,eig_vectors_n, "LDA", rand_idx ,all_mean,train_label)

    return eig_vectors_n

def plot_confusion_matrix(cm, classes, cmap):
    
    plt.figure(figsize = (10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(int(cm[i, j]), 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')

def metrics(prediction,test_y):
    res = np.zeros((15,15))
    for i in range(len(test_y)):
        res[int(test_y[i])-1,int(prediction[i])-1] +=1
    plot_confusion_matrix(res, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], cmap="BuPu")

def KNN(train_X,train_y,test_X,test_y,type,k=3):
    prediction = np.zeros(test_X.shape[0])
    correct = 0
    for i in range(test_X.shape[0]):
        dis_matrix = distance.cdist(test_X[i].reshape(1,-1), train_X, 'euclidean').reshape(-1)
        idx = np.argsort(dis_matrix)[:k]
        prediction[i] = np.argmax(np.bincount(train_y[idx].astype('int64')))
        if(prediction[i] == test_y[i]):
            correct+=1
    
    metrics(prediction,test_y)
    plt.savefig(f'./cm/cm-{type}-{k}.png')
    print(f'Accuracy = {correct/test_X.shape[0]}')

def rbf(data_1,data_2,gamma=1e-5):
    ret = np.exp((-gamma**2) * distance.cdist(data_1,data_2, 'sqeuclidean'))
    return ret

def sigmoid(data_1,data_2,gamma=1e-5,coef=1e-5):
    ret = np.tanh(gamma * np.dot(data_1,data_2.T) + coef)
    return ret

def poly(data_1, data_2, gamma=1e-6, coef=1e-2, degree=2):
    ret = (gamma * np.dot(data_1,data_2.T) + coef) ** degree
    return ret

def kernelPCA(train_data, rand_idx , kernel="sigmoid"):
    # calculate mean_face
    mean = train_data.mean(axis = 0)
    train_data_scaled = train_data - mean

    if (kernel == "rbf"):
        gram_matrix = rbf(train_data.T,train_data.T)
    elif (kernel == "sigmoid"):
        gram_matrix = sigmoid(train_data.T,train_data.T)
    else:
        gram_matrix = poly(train_data.T,train_data.T)
    
    n_1 = np.ones((gram_matrix.shape[0],gram_matrix.shape[0])) * (1/gram_matrix.shape[0])
    gram_matrix_ = gram_matrix - np.dot(n_1,gram_matrix) - np.dot(gram_matrix,n_1) + np.dot(np.dot(n_1,gram_matrix),n_1)
    # solve eigen_problem
    eig_values, eig_vectors = np.linalg.eigh(gram_matrix_)
    # pick 25 eigen_vectors
    idx = eig_values.argsort()[::-1][:N_component]
    eig_vectors_n = eig_vectors[:,idx].astype(np.float64)

    show_eig_face(eig_vectors_n,f'Kernel_{kernel}_PCA')
    reconstruct(train_data,train_data_scaled,eig_vectors_n, f'Kernel_{kernel}_PCA', rand_idx ,mean)
    
    return eig_vectors_n

def kernelLDA(train_data, train_label, rand_idx, kernel="sigmoid"):

    if (kernel == "rbf"):
        gram_matrix = rbf(train_data.T,train_data.T)
    elif (kernel == "sigmoid"):
        gram_matrix = sigmoid(train_data.T,train_data.T)
    else:
        gram_matrix = poly(train_data.T,train_data.T)

    #S_w and S_b
    label_mean =np.zeros((15,RESIZE[0]*RESIZE[1]))
    S_w = np.zeros((RESIZE[0]*RESIZE[1],RESIZE[0]*RESIZE[1]))
    S_b = np.zeros((RESIZE[0]*RESIZE[1],RESIZE[0]*RESIZE[1]))
    all_mean = np.mean(gram_matrix,axis=1)

    for i in range(15):
        idx = np.where(train_label == i+1)[0]
        t = np.identity(len(idx)) * (1/len(idx))
        S_w += np.dot(np.dot(gram_matrix[:,idx],t),gram_matrix[:,idx].T)
        
        label_mean[i,:] = np.mean(gram_matrix[idx,:],axis=0)
        bi = (label_mean[i,:] - all_mean).reshape(-1,1)
        S_b += len(idx) * np.dot(bi,bi.T)

    # pseudo-inverse
    matrix = np.dot(np.linalg.pinv(S_w),S_b)


    eig_values, eig_vectors = np.linalg.eigh(matrix)
    idx = eig_values.argsort()[::-1][:N_component]
    eig_vectors_n = eig_vectors[:,idx].astype(np.float64)

    show_eig_face(eig_vectors_n,f'Kernel_{kernel}_LDA')
    train_data_scaled = train_data-all_mean
    reconstruct(train_data,train_data_scaled,eig_vectors_n, f'Kernel_{kernel}_LDA', rand_idx ,all_mean,train_label)

    return eig_vectors_n

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", default=1, type=int)
    args = parser.parse_args()

    RESIZE = [70,60]  
    N_component = 25
    random_n = 10
    # (135, RESIZE[0]*RESIZE[1]), (30, RESIZE[0]*RESIZE[1])
    train_data, test_data,train_label,test_label = load_data()
    # random picking
    rand_idx = np.random.choice(test_data.shape[0],random_n,replace=False)
    #rand_idx = np.array([15,17,0,29,11,24,13,3,21,9])

    if(args.q == 1):
        eig_vectors_PCA = PCA(train_data, rand_idx)
        eig_vectors_LDA = LDA(train_data, train_label, rand_idx)

    if(args.q == 2):
        mean = np.mean(train_data,axis=0)

        eig_vectors_PCA = PCA(train_data, rand_idx)
        train_X_PCA = np.dot((train_data - mean),eig_vectors_PCA)
        test_X_PCA = np.dot((test_data-mean),eig_vectors_PCA)
        
        eig_vectors_LDA = LDA(train_data, train_label, rand_idx)
        train_X_LDA = np.dot((train_data-mean),eig_vectors_LDA)
        test_X_LDA = np.dot((test_data-mean),eig_vectors_LDA)

        k=[3,5,10,20]
        for i in range(len(k)):
            print("PCA_",k[i])
            KNN(train_X_PCA,train_label,test_X_PCA,test_label,"PCA",k[i])
            print("LDA_",k[i])
            KNN(train_X_LDA,train_label,test_X_LDA,test_label,"LDA",k[i])

    if(args.q == 3):
        mean = np.mean(train_data,axis=0)
        train_y = train_label
        test_y = test_label
        kernel= ['sigmoid','rbf','poly']
        k=[3,5,10,20]
        for i in range(len(k)):
            for j in range(len(kernel)):
                eig_vectors_PCA = kernelPCA(train_data, rand_idx,kernel[j])
                train_X_PCA = np.dot((train_data-mean),eig_vectors_PCA)
                test_X_PCA = np.dot((test_data-mean),eig_vectors_PCA)
                eig_vectors_LDA = kernelLDA(train_data, train_label, rand_idx, kernel[j])
                train_X_LDA = np.dot((train_data-mean),eig_vectors_LDA)
                test_X_LDA = np.dot((test_data-mean),eig_vectors_LDA)
                KNN(train_X_PCA,train_y,test_X_PCA,test_y,f'Kernel_{kernel[j]}_PCA',k[i])
                KNN(train_X_LDA,train_y,test_X_LDA,test_y,f'Kernel_{kernel[j]}_LDA',k[i])

        