#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score


# In[41]:


X=np.random.randn(2,100)
y=np.random.randn(3,100)


# In[42]:


def initialisation(dimensions):   #liste des dimensions des couches
    
    parametres = {}  # dictionnaire de parametres
    C = len(dimensions) # nombre de couches du réseau
    print('dim = ', dimensions)
    for c in range(1, C): # boucle sur les couches
        parametres['W'+ str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parametres['b'+ str(c)] = np.random.randn(dimensions[c], 1)
        
    return parametres


# In[43]:


# parametres = initialisation ([2,16,16,3])

# for key, val in parametres.items():
#     print(key, val.shape)


# In[44]:


def forward_propagation(X, parametres):
    
    activations = {'A0' : X}
    C = len(parametres) // 2
    
    
    for c in range(1, C + 1):
        Wc = parametres['W' + str(c)]
        Acm1 = activations['A' + str(c - 1)]
        bc = parametres['b' + str(c)]
        Z = Wc.dot(Acm1) + bc
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z)) 

    return activations


# In[45]:


# activations = forward_propagation(X, parametres)
#
# for key, val in activations.items():
#     print(key, val.shape)


# In[46]:


def back_propagation(y, activations, parametres):
    
    m = y.shape[1]
    C = len(parametres) // 2
    
    dZ = activations['A' + str(C)] - y
    gradients = {}
    
    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)]) 
        
    return gradients
    


# In[47]:


# grad = back_propagation(y, activations, parametres)
#
# for key, val in grad.items():
#     print(key, val.shape)
#

# In[48]:


def update(gradients, parametres, learning_rate):
    
    C = len(parametres) // 2
    
    for c in range(1, C + 1):
        parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]
    
    return parametres


# In[ ]:


def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    C = len(parametres) // 2
    Af = activations['A' + str(C)]

    return Af


# In[50]:

def log_loss(A, y):
    LL = 0
    eps = 1e-12
    for i in range(len(y)) :
        LL += 1 / len(y[i]) * np.sum(-y[i] * np.log(A[i]+eps) - (1 - y[i]) * np.log(1 - A[i]+eps))
    return LL


def neural_network(X, y, hidden_layers = (32,32,32), learning_rate = 0.1, n_iter = 1000):
    
    np.random.seed(0)
    # initialisation
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    parametres = initialisation(dimensions)
    
    train_loss = []
    train_acc = []
    
    for i in tqdm(range(n_iter)):

        activations = forward_propagation(X, parametres)
        gradients = back_propagation(y, activations, parametres)
        parametres = update(gradients, parametres, learning_rate)
        
        if i %10 == 0:
            C = len(parametres) // 2
            train_loss.append(log_loss(y, activations['A' + str(C)]))
            y_pred = predict(X, parametres)
            current_accuracy = r2_score(y.flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)
            
    # Visualization
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 4))
    ax[0].plot(train_loss, label='train loss')
    ax[0].legend()
    
    ax[1].plot(train_acc, label='train acc')
    ax[1].legend()
    # visualisation(X, y, parametres, ax)
    plt.show()
    
    return parametres, current_accuracy
            


# In[ ]:




