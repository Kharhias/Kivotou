#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score


# In[22]:


# X, y = make_blobs(n_samples = 100, n_features = 5, centers = 2, random_state = 0)
# y = y.reshape((y.shape[0], 1))

# print('dimensions de X:', X.shape)
# print('dimensions de y:', y.shape)
#
# plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
# plt.show()
#

# In[23]:


def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


# In[4]:


def model(X, W, b):
    Z=0
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A


# In[5]:


def log_loss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))


# In[6]:


def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)


# In[7]:


def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)


# In[17]:


def predict(X, W, b):
    A = model(X, W, b)
    # print(A.shape)
    return A #>= 0.5


# In[18]:


from sklearn.metrics import accuracy_score


# In[19]:


def artificial_neuron(X, y, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    W, b = initialisation(X)
    Loss = []
    
    for i in tqdm(range(n_iter)):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

        
    y_pred = predict(X, W, b)
    print("précision : ", r2_score(y, y_pred))

    plt.plot(Loss)
    plt.show()
    
    return (W, b)



