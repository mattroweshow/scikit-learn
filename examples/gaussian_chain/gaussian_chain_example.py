__author__ = 'rowem'

import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_mldata
from gaussian_chain import SingleGaussianChain, DoubleGaussianChain
from sklearn.naive_bayes import MultinomialNB

# From remote server
custom_data_home = "/home/rowem/Documents/Git/Data/scikit-minst"
# data = fetch_mldata('news20.binary', data_home=custom_data_home)
data = fetch_mldata('covtype.binary_scale', data_home=custom_data_home)

# Use only the first 1000 instances for now
instance_ids = range(0,1000)

# Only use the first 3 features for now
# data_X = data.data[[instance_ids],[feature_ids]]
data_X = data.data[instance_ids,:]

data_y = data.target[instance_ids]
# Relabel the y values to be 0 or 1
data_y_b = np.mod(1, data_y)
data_y = data_y_b



# target_class = next(iter(set(data_y)))
target_class = 1


# Prepare it for experiments
np.random.seed(0)
indices = np.random.permutation(data_X.shape[0])

# Get the 90% cutoff in terms of absolute indices
cutoff = int(data_X.shape[0] * 0.1)
# Randomly split into training and testing
X_train = data_X[indices[:-cutoff]]
y_train = data_y[indices[:-cutoff]]
X_test = data_X[indices[-cutoff:]]
y_test = data_y[indices[-cutoff:]]

# Set the hyperparameters
eta = 0.001
lambdA = 0.001
# rho = 0.9
alpha = 0.5
learning_mode = 2

print "Naive Bayes"
clf = MultinomialNB().fit(X_train, y_train)
# predicted_nb = clf.predict_proba(X_test)
predicted_nb = clf.predict(X_test)
print(metrics.roc_auc_score(y_test, predicted_nb))

# Train the model
print "Double Gaussian Chain"
rhos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for rho in rhos:
    print str(rho)
    # print data_X.shape
    gcm = DoubleGaussianChain(rho, alpha, lambdA, eta, learning_mode, target_class)
    gcm.fit(X_train, y_train)
    predicted_gcm = gcm.predict_proba(X_test)
    print(metrics.roc_auc_score(y_test, predicted_gcm))

# print "Single Gaussian Chain"
# rhos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# for rho in rhos:
#     print str(rho)
#     # print data_X.shape
#     gcm = SingleGaussianChain(rho, alpha, lambdA, eta, learning_mode, target_class)
#     gcm.fit(X_train, y_train)
#     predicted_gcm = gcm.predict_proba(X_test)
#     print(metrics.roc_auc_score(y_test, predicted_gcm))

