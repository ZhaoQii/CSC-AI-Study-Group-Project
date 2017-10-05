# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:17:34 2016

@author: ap
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
#from sklearn.cluster import KMeans
#from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from pylab import *


#cluster_num = 10

#kmeans = KMeans(n_clusters=cluster_num)
#kmeans.fit(X)

#centroids = kmeans.cluster_centers_
#labels = kmeans.labels_

#print("centroids : ")
#print(centroids)
#print("labels : ")
#print(labels)
def ThreeD_Visualization(X, real_labels):
    colors = ['g','r','c','y','b','m','black','purple','orange','brown']

    #color = np.random.rand(cluster_num)

    #c = Counter(labels)
    fig = figure()
    ax = fig.gca(projection='3d')

    for i in range(len(X)):
        #print("coordinate:",X[i], "label:", labels[i])
        #print("i : ",i)
        #print("color[labels[i]] : ",colors[labels[i]])
        label = np.int(real_labels[i])
        ax.scatter(X[i,0], X[i,1], X[i,2], c=colors[label])

    #for cluster_number in range(cluster_num):
        #print("Cluster {} contains {} samples".format(cluster_number, c[cluster_number]))

    #ax.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2], marker = "x", s=150, linewidths = 5, zorder = 100)

    plt.show()