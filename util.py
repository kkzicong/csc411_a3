import scipy.io as io
import numpy as np
from sklearn import cross_validation
from random import shuffle as shuffle1
from sklearn.utils import shuffle as shuffle2
import matplotlib.pyplot as plt
from skimage import exposure
from sklearn import preprocessing


def write_results(labels, file):
    """
    Given a set of labels and a file name and produce an output file which we can submit
    to the kaggle website.
    """
    with open(file, 'w') as out:
        out.write("Id,Prediction\n")
        for i, l in enumerate(labels):
            out.write(','.join([str(i+1),str(l)])+"\n")
        if len(labels) < 1253:
            for i in range(len(labels), 1253):
                out.write(','.join([str(i+1),"0"])+"\n")


def load_labeled_data():
    """load labeled data"""
    
    load_labeled = io.loadmat("/home/zi/Downloads/data/labeled_images.mat")
   
    images = load_labeled['tr_images']   
    labels = load_labeled['tr_labels']
    identity = load_labeled['tr_identity']
   
    x, y, n = images.shape
    N, M = n, x * y  

    images = images.reshape(M, N).T
    
# flatten the data    
    labels = labels.flatten()
    identity = identity.flatten()

    return images,labels, identity

def load_public_test():
     """load public test images"""
     load_test = io.loadmat("/home/zi/Downloads/data/public_test_images.mat")
     
     test = load_test['public_test_images']
     x,y,n=test.shape
     N,M = n, x*y
     
     test = test.reshape(M, N).T

     return test

def load_unlabeled_data():
     """load unlabeled data"""
     load_unlabeled = io.loadmat("/home/zi/Downloads/data/unlabeled_images.mat")
     images = load_unlabeled['unlabeled_images'] 
     
     x, y, n = images.shape
     N, M = n, x * y
     images = images.reshape(M, N).T 

     return images

def load_hidden_test():
     """load hidden test images"""
     load_unlabeled = io.loadmat("/home/zi/Downloads/data/hidden_test_images.mat")
     images = load_unlabeled['hidden_test_images'] 
     
     x, y, n = images.shape
     N, M = n, x * y
     images = images.reshape(M, N).T 

     return images

def train_test_split(data, target, identity):
    """split the data into trainning set and test set."""
    data_dic = {}
    target_dic = {}
    
    for i in range(len(data)):
        if identity[i] in data_dic:
            data_dic[identity[i]] = np.concatenate((data_dic[identity[i]], [data[i]]), axis=0)
            target_dic[identity[i]] = np.concatenate((target_dic[identity[i]], [target[i]]), axis=0)
        else:
            data_dic[identity[i]] = [data[i]]
            target_dic[identity[i]] = [target[i]]
    
    unid_data = data_dic[-1]
    unid_target = target_dic[-1]
    
    n = len(target) * 0.7
    
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(unid_data, unid_target, test_size=0.24)

    del data_dic[-1]
    del target_dic[-1]
    
    items = target_dic.items()
    shuffle1(items)
    
    for key, value in items:
        if (len(y_train) < n):
            x_train = np.concatenate((x_train, data_dic[key]), axis=0)
            y_train = np.concatenate((y_train, value), axis=0)
        else:
            x_test = np.concatenate((x_test, data_dic[key]), axis=0)
            y_test = np.concatenate((y_test, value), axis=0)

    x_train, y_train = shuffle2(x_train, y_train)
    x_test, y_test = shuffle2(x_test, y_test)

    return x_train, x_test, y_train, y_test
    
def equalize(inputs):
    """equalize the data"""
    new_data = []
    for i in inputs:
        new_i = exposure.equalize_hist(i)
        new_data.append(new_i)
    return np.array(new_data)

def gabor_filter(inputs, theta, sigma, frequency):
    from skimage.filters import gabor_kernel
    from scipy import ndimage as ndi
    new_data = []
    kernel = np.real(gabor_kernel(frequency=frequency, theta=theta, sigma_x=sigma,sigma_y=sigma))
    for i in inputs:
        i = np.reshape(i,(32,32))
        filtered = ndi.convolve(i, kernel, mode='wrap')
        new_data.append(filtered.reshape(1024))
    return np.array(new_data)

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(len(images)):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def scale(inputs):
    """standardize the data"""
    new_data = []
    for i in inputs:
        new_i = preprocessing.scale(i)
        new_data.append(new_i)
    return np.array(new_data)

def normalize(inputs):
    """normalize the data"""
    new_data = []
    for i in inputs:
        new_i = preprocessing.normalize(i, norm='l2')
        new_data.append(new_i[0])
    return np.array(new_data)
    