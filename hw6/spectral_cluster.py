import numpy as np
import scipy.spatial.distance as distance
from PIL import Image
import random
import argparse
import matplotlib.pyplot as plt
import os 

def load_data(image):
    im = Image.open(image)
    data = np.array(im).reshape(10000,3)
    spatial = []
    for i in range(100):
        for j in range(100):
            spatial.append([i,j])
    spatial = np.array(spatial)
    return data,spatial

def kernel(spatial_A, spatial_B, rgb_A, rgb_B, s=1e-4, c=1e-4):
    s_dis = distance.cdist(spatial_A, spatial_B, 'sqeuclidean')
    c_dis = distance.cdist(rgb_A, rgb_B, 'sqeuclidean')
    val = np.exp((-s * s_dis)+(-c * c_dis))
    return val

def graph_laplacian(kernel_val,cut):
    delta = np.identity(len(kernel_val))
    D = np.sum(kernel_val,axis=0) * delta
    W = kernel_val
    L = D - W
    if cut =="normalize":
        laplacian = np.dot(np.dot(D**0.5,L),D**0.5)
    if cut == "ratio":
        laplacian = L
    return laplacian

def init_cluster(k,data,method,kernel_val):
    if method == "r":
        return np.random.randint(k, size=data.shape[0])
    elif method == "p":
        seed = []
        seed.append(np.random.randint(0,data.shape[0]))
        diag = kernel_val.diagonal().reshape(-1, 1)
        distance = diag - 2 * kernel_val + diag
        for center in range(k-1):
            dist = np.min(distance[seed], axis=0)
            p = dist/sum(dist)
            seed.append(np.random.choice(data.shape[0], 1, p=list(p))[0])
        init_cluster_res = np.argmin(distance[seed], axis=0)
        return init_cluster_res   

def find_center(cluster_res,spatial,data,k):
    center_pixel = np.zeros((k,spatial.shape[1]))
    center_data = np.zeros((k,data.shape[1]))
    for i in range(k):
        center_pixel[i] = np.mean(spatial[np.where(cluster_res==i)[0]],axis=0)
        center_data[i] = np.mean(data[np.where(cluster_res==i)[0]],axis=0)
    return center_pixel,center_data

def visualize(total_cluster_res,spatial,data,args,change_list):
    k = args.k
    if(args.m == 'p'):
        method = f'{args.c}_kmean++'
    else:
        method = f'{args.c}_random'
    file = args.d
    folder = f'{file.split(".")[1][1:]}_{k}_{method}'
    os.mkdir(f'./figure/{folder}')

    for i in range(len(total_cluster_res)):
        center_pixel,center_rgb = find_center(total_cluster_res[i],spatial,data,k)
        plt.figure()
        for j in range(k):
            plt.scatter(spatial[total_cluster_res[i]==j][:,0], spatial[total_cluster_res[i]==j][:,1], color = center_rgb[j]/255,s=1)
        for j in range(k):
            plt.scatter(center_pixel[j,0], center_pixel[j,1], color = center_rgb[j]/255,edgecolors='black')
        plt.title(f'method:{method},iter:{i+1},k:{k}')
        plt.savefig(f'./figure/{folder}/iter{i+1}.png')

    plt.figure()
    x = [j for j in range(len(change_list))]
    plt.plot(x,change_list)
    plt.title(f'method:{method},iter:{i+1},k:{k}, distance_change_history')
    plt.savefig(f'./figure/{folder}/change.png')

def visualize_eigenspace(total_cluster_res,T,data,args):
    k = args.k
    if(args.m == 'p'):
        method = f'{args.c}_eigenspace_kmean++'
    else:
        method = f'{args.c}_eigenspace_random'
    file = args.d
    folder = f'{file.split(".")[1][1:]}_{k}_{method}'
    os.mkdir(f'./figure/{folder}')

    center_data,center_rgb = find_center(total_cluster_res[-1],T,data,k)
    plt.figure()
    for j in range(k):
        plt.scatter(T[total_cluster_res[-1]==j][:,0], T[total_cluster_res[-1]==j][:,1], color = center_rgb[j]/255,s=1)
    for j in range(k):
        plt.scatter(center_data[j,0], center_data[j,1], color = center_rgb[j]/255,edgecolors='black')
    plt.title(f'method:{method},k:{k}')
    plt.savefig(f'./figure/{folder}/eigenspace.png')

def kmean(data,T,spatial,args,kernel_val):
    k = args.k 
    method = args.m
    total_cluster_res = []
    init_cluster_res = init_cluster(k,data,method,kernel_val)
    total_cluster_res.append(init_cluster_res)
    cluster_res = init_cluster_res
    center_pixel,center_T = find_center(cluster_res,spatial,T,k)
    old_distance_k = np.zeros((k,len(data)))
    count = 0
    change_list = []
    for iter in range(100):
        distance_k  = np.zeros((k,len(data)))
        for i in range(k):
             distance_k[i] = distance.cdist(T, center_T[i].reshape(1,-1), 'euclidean').reshape(-1)
        cluster_res = np.argmin(distance_k, axis=0)
        total_cluster_res.append(cluster_res)
        count += 1
        change = np.sum(abs(old_distance_k-distance_k))
        change_list.append(change)
        if(change < 1e-50):
            break
        else:
            print(f'iter:{count}, {np.sum(abs(old_distance_k-distance_k))}')
            old_distance_k = distance_k
            center_pixel,center_T = find_center(cluster_res,spatial,T,k)

    visualize(total_cluster_res,spatial,data,args,change_list)
    visualize_eigenspace(total_cluster_res,T,data,args)

def cal_eig_vecs(k,laplacian,cut):
    eig_value, eig_vector = np.linalg.eigh(laplacian)
    sort_perm = eig_value.argsort()[1:k+1]
    if cut == "normalize":
        U = eig_vector[:, sort_perm]
        norm_1 = (np.sum(U**2,axis=1)**0.5).reshape(-1,1)
        T = U / norm_1
    else:
        T = eig_vector[:, sort_perm]
    return T

def spectral_cluster(data,spatial,args):
    k = args.k 
    method = args.m
    cut = args.c
    kernel_val = kernel(spatial, spatial, data, data)
    laplacian = graph_laplacian(kernel_val,cut)
    T = cal_eig_vecs(k,laplacian,cut)
    kmean(data,T,spatial,args,kernel_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", default="./image1.png", type=str)
    parser.add_argument("-k", default=3, type=int)
    parser.add_argument("-m", default="r", type=str)
    parser.add_argument("-c", default="normalize", type=str)
    args = parser.parse_args()
    data,spatial = load_data(args.d)
    spectral_cluster(data,spatial,args)
    print("Finish!!")