
import numpy as np
from scipy.spatial import distance
import scipy

def EuclideanDistances(A):       #calculate the euclidean distance of A's element
    
    dismat = distance.pdist(A, 'sqeuclidean')   
    dismat = distance.squareform(dismat)
    
    return np.matrix(dismat)

# Define the function to return the specific dimension representation of dataset by Spectral Clustering
# Sigma is the hyperparameter controling the measurement of similarity
def getSCpresentation(dataset,dimension,sigma):    #dataset should not involve real labels
    #dimension here should be 10 as we wanted
    #for sigma I use 300 after some trials
    
    dist = EuclideanDistances(dataset)
    similarity = scipy.exp(-(dist ** 2)  /(2 * sigma ** 2))  # get the similarity difined by Spectral Clustering
    similarity = similarity-np.diag(np.diag(similarity))
    D = np.diagflat(np.transpose(np.array(similarity.sum(axis=1)) ** -.5))
    L = np.dot(np.dot(D,similarity),D)
    eig_vals,eig_vecs = np.linalg.eig(L)
    idx = eig_vals.argsort()[::-1]   
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:,idx]
    X = eig_vecs[:,0:dimension].real
    return X
