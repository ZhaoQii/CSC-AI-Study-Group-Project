# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:45:12 2016

@author: ap
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 23:13:32 2016

@author: QI ZHAO
"""
import numpy as np
from scipy.spatial import distance
import scipy

def EuclideanDistances(A):       #calculate the euclidean distance of A's element
    
    dismat = distance.pdist(A, 'sqeuclidean')   
    dismat = distance.squareform(dismat)
    
    return np.matrix(dismat)


def getSCpresentation(dataset,dimension,sigma):    #dataset should not involve real labels
    #dimension here should be 10 as we wanted
    #for sigma I use 300
    #
    dist = EuclideanDistances(dataset)
    similarity = scipy.exp(-(dist ** 2)  /(2 * sigma ** 2))
    similarity = similarity-np.diag(np.diag(similarity))
    D = np.diagflat(np.transpose(np.array(similarity.sum(axis=1)) ** -.5))
    L = np.dot(np.dot(D,similarity),D)
    eig_vals,eig_vecs = np.linalg.eig(L)
    idx = eig_vals.argsort()[::-1]   
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:,idx]
    X = eig_vecs[:,0:dimension].real
    return X
