import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

from mpl_toolkits.mplot3d import Axes3D
from pylab import *

def ThreeD_Visualization(X, real_labels):
    colors = ['g','r','c','y','b','m','black','purple','orange','brown']

    fig = figure()
    ax = fig.gca(projection='3d')

    for i in range(len(X)):
        label = np.int(real_labels[i])
        ax.scatter(X[i,0], X[i,1], X[i,2], c=colors[label])
    plt.show()
