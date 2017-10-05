# CSC-AI-Study-Group-Project   
### Qi Zhao qz2316
Hi everyone this is my project for AI study group. Here I set up a two hidden layers network for MNIST recognition.   
* In lib, you could find a python file named 'MNIST Recognition with TF.py' which is the main code for MNIST training and prediction. And for checking the results you may go to doc and there is a file named 'Train and Prediction Result.ipynb' and it contains the training cost each epoch and the final prediction accuracy. While I have to say this is a quite easy model, the performance is still great, which is around 98% in testing set.   
* Then I want to apply dimensionality deduction method to extract certain features from the original data input. And after obtaining low dimensional features, we have the new input data which may save our training time and improve the prediction performance. This is the reason Spectral Clustering here, and the other files in lib are some functions for extracting low d features using SC and visualization(3d). The figs contains two graphs (of a only small batch of dataset to make it computable) showing the real labels and K-Means labels (just using 'sklearn.cluster's 'KMeans' function) in 3d. But when I tried to input the 10 d features into network and executed what I did for original data, I found the training cost declined very slowly and converged to a large value. I am still working on solving it.   
