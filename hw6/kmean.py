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
    s_dis = distance.cdist(spatial_A, spatial_B, 'euclidean')**2
    c_dis = distance.cdist(rgb_A, rgb_B, 'euclidean')**2
    val = np.exp((-s * s_dis)+(-c * c_dis))
    return val

def init_cluster(k,data,method,kernel_val):
    if method == "r":
        return np.random.randint(k, size=data.shape[0])
    elif method == "p":
        seed = []
        random_point = np.random.randint(0,data.shape[0])
        seed.append(random_point)
        diag = kernel_val.diagonal()     
        distance_to_p = diag.reshape(-1, 1) - 2 * kernel_val + diag.reshape(-1, 1)
        for center in range(k-1):
            dist = np.min(distance_to_p[seed], axis=0)
            p = dist/dist.sum()
            new_seed = p.argmax()
            seed.append(new_seed)
            
        init_cluster_res = np.argmin(distance_to_p[seed], axis=0)
        return init_cluster_res    
                
def calculate_dist(kernel_val,cluster_res,k,data,i):
    cluster_member = np.where(cluster_res == i)[0]
    member_boolean = np.zeros(len(data))
    member_boolean[cluster_member] = 1
    one = kernel_val.diagonal()
    second = 2 / len(cluster_member) * np.dot(kernel_val.T, member_boolean)
    val = 0
    for p in cluster_member:
        for q in cluster_member:
            val += kernel_val[p,q]
    three = (1 / len(cluster_member)**2)*val
    dist = one - second + three

    return dist

def find_center(cluster_res,spatial,data,k):
    center_pixel = np.zeros((k,2))
    center_rgb = np.zeros((k,3))
    for i in range(k):
        center_pixel[i] = np.mean(spatial[np.where(cluster_res==i)[0]],axis=0)
        center_rgb[i] = np.mean(data[np.where(cluster_res==i)[0]],axis=0)
    return center_pixel,center_rgb

def visualize(total_cluster_res,spatial,data,args,change_list):
    k = args.k
    if(args.m == 'p'):
        method = "kmean++"
    else:
        method = "random"
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


def kernel_kmean(data,spatial,args):
    k = args.k 
    method = args.m
    kernel_val = kernel(spatial, spatial, data, data)
    total_cluster_res = []
    init_cluster_res = init_cluster(k,data,method,kernel_val)
    total_cluster_res.append(init_cluster_res)
    cluster_res = init_cluster_res
    old_distance_k = np.zeros((k,len(data)))
    count = 0
    change_list = []
    for iter in range(100):
        distance_k  = np.zeros((k,len(data)))
        for i in range(k):
             distance_k[i] = calculate_dist(kernel_val,cluster_res,k,data,i)
        cluster_res = np.argmin(distance_k, axis=0)
        total_cluster_res.append(cluster_res)
        count += 1
        change = np.sum(abs(old_distance_k-distance_k))
        change_list.append(change)
        print(f'iter:{count}, {change}')
        if(change < 1e-50):
            break
        else:
            old_distance_k = distance_k

    visualize(total_cluster_res,spatial,data,args,change_list)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", default="./image1.png", type=str)
    parser.add_argument("-k", default=3, type=int)
    parser.add_argument("-m", default="r", type=str)
    args = parser.parse_args()
    data,spatial = load_data(args.d)
    kernel_kmean(data,spatial,args)
    print("Finish!!")