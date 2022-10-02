import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image as Image

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, algo="t_SNE"):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 300
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    Y_history = np.zeros((max_iter,n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        
        # different Q
        if(algo == "t_SNE"):
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        elif (algo =="s_SNE"):
            num = np.exp(-(1. + np.add(np.add(num, sum_Y).T, sum_Y)))

        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient (different gradient)
        PQ = P - Q
        for i in range(n):
            if(algo == "t_SNE"):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            elif (algo =="s_SNE"):
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            
        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
        Y_history[iter,:,:] = Y

    # Return solution
    return Y_history, P, Q

def load_data():
    X = np.loadtxt("./tsne_python/mnist2500_X.txt")
    labels = np.loadtxt("./tsne_python/mnist2500_labels.txt")
    
    return X,labels

def visualize(Y_history, labels, perplexity, algo):

    if not os.path.isdir(f'./SNE/{algo}_{perplexity}/'):
        os.mkdir(f'./SNE/{algo}_{perplexity}/')
        os.mkdir(f'./SNE/{algo}_{perplexity}/gif/')
        os.mkdir(f'./SNE/{algo}_{perplexity}/dis/')

    cmap =['red','orange','blue','gray','purple','yellow','green','pink','lightskyblue','springgreen']
    label_unique = np.unique(labels)
    

    for i in range(Y_history.shape[0]):
        plt.figure(figsize = (10,10))
        for j in range(len(label_unique)):
            plt.title(f'{algo}-{perplexity}_{i}')
            idx = np.where(labels == label_unique[j])[0]
            plt.scatter(Y_history[i,idx,0],Y_history[i,idx,1],color = cmap[j],label= j)
            plt.legend()
            
        plt.savefig(f'./SNE/{algo}_{perplexity}/{algo}-{perplexity}_{i}.png')

    # plot gif
    imgs = []
    for i in range(Y_history.shape[0]):
        temp = Image.open(f'./SNE/{algo}_{perplexity}/{algo}-{perplexity}_{i}.png')
        imgs.append(temp)
    save_name = f'./SNE/{algo}_{perplexity}/gif/{algo}_{perplexity}.gif'
    imgs[0].save(save_name, save_all=True, append_images=imgs, duration=10)

def plot_dis(P, Q, perplexity, algo):
    P = P.reshape(-1)
    Q = Q.reshape(-1)
    min_P_idx = np.where(P == np.min(P))[0]
    min_Q_idx = np.where(Q == np.min(Q))[0]
    P = np.delete(P,min_P_idx)
    Q = np.delete(Q,min_Q_idx)
    log_P = np.log(P)
    log_Q = np.log(Q)
    
    plt.figure(figsize = (10,20))
    fig, ax = plt.subplots(2)

    ax[0].hist(log_P, bins=200)
    ax[0].set_title(f'{algo}-{perplexity}_high-dimensional space')
    ax[1].hist(log_Q, bins=200)
    ax[1].set_title(f'{algo}-{perplexity}_low-dimensional space')
    
    plt.subplots_adjust(hspace=1)
    plt.savefig(f'./SNE/{algo}_{perplexity}/dis/{algo}-{perplexity}_dis.png')

if __name__ == "__main__":
    X,labels = load_data()
    perplexity = [5,15,30,50]
    algo = ["t_SNE","s_SNE"]
    for i in range(len(perplexity)):
        for j in range(len(algo)):
            print(f'{algo[j]}_{perplexity[i]}')
            Y_history, P, Q = tsne(X, 2, 50, perplexity[i], algo[j])
            visualize(Y_history, labels, perplexity[i], algo[j])
            plot_dis(P, Q, perplexity[i], algo[j])
            plt.close('all')
    '''perplexity = 15
    algo = "s_SNE"
    Y_history, P, Q = tsne(X, 2, 50, perplexity, algo)
    #visualize(Y_history, labels, perplexity, algo)
    print(Q)
    plot_dis(P, Q, perplexity, algo)'''