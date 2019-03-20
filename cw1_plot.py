
# coding: utf-8

# In[155]:


import numpy as np
import matplotlib.pyplot as plt


# In[156]:


iris = np.loadtxt('iris.data',str,delimiter=',')
[rows,cols]=iris.shape
print(iris.dtype)
print(float(iris[1][1]))


# In[158]:


fig = plt.figure()
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
axes[0][0].set_xlabel('petal width(cm)')
axes[0][0].set_ylabel('petal length(cm)')
#axes[0][0].legend(loc='upper left')
for i in range(rows):
    if iris[i][4] == 'Iris-setosa':
        axes[0][0].scatter(float(iris[i][3]),float(iris[i][2]),c='blue',marker='s')
    if iris[i][4] == 'Iris-versicolor':
        axes[0][0].scatter(float(iris[i][3]),float(iris[i][2]),c='red',marker='^')
    if iris[i][4] == 'Iris-virginica':
        axes[0][0].scatter(float(iris[i][3]),float(iris[i][2]),c='green',marker='o')

axes[0][1].set_xlabel('sepal width(cm)')
axes[0][1].set_ylabel('sepal length(cm)')
#axes[0][1].legend(loc='upper left')
for i in range(rows):
    if iris[i][4] == 'Iris-setosa':
        axes[0][1].scatter(float(iris[i][1]),float(iris[i][0]),c='blue',marker='s')
    if iris[i][4] == 'Iris-versicolor':
        axes[0][1].scatter(float(iris[i][1]),float(iris[i][0]),c='red',marker='^')
    if iris[i][4] == 'Iris-virginica':
        axes[0][1].scatter(float(iris[i][1]),float(iris[i][0]),c='green',marker='o')

axes[1][0].set_xlabel('sepal width(cm)')
axes[1][0].set_ylabel('petal width(cm)')
#axes[1][0].legend(loc='upper left')
for i in range(rows):
    if iris[i][4] == 'Iris-setosa':
        axes[1][0].scatter(float(iris[i][1]),float(iris[i][3]),c='blue',marker='s')
    if iris[i][4] == 'Iris-versicolor':
        axes[1][0].scatter(float(iris[i][1]),float(iris[i][3]),c='red',marker='^')
    if iris[i][4] == 'Iris-virginica':
        axes[1][0].scatter(float(iris[i][1]),float(iris[i][3]),c='green',marker='o')
        
axes[1][1].set_xlabel('sepal length(cm)')
axes[1][1].set_ylabel('petal length(cm)')
#axes[1][1].legend(loc='upper left')
for i in range(rows):
    if iris[i][4] == 'Iris-setosa':
        axes[1][1].scatter(float(iris[i][0]),float(iris[i][2]),c='blue',marker='s')
    if iris[i][4] == 'Iris-versicolor':
        axes[1][1].scatter(float(iris[i][0]),float(iris[i][2]),c='red',marker='^')
    if iris[i][4] == 'Iris-virginica':
        axes[1][1].scatter(float(iris[i][0]),float(iris[i][2]),c='green',marker='o')
        
axes[2][1].set_xlabel('sepal length(cm)')
axes[2][1].set_ylabel('petal width(cm)')
#axes[2][1].legend(loc='upper left')
for i in range(rows):
    if iris[i][4] == 'Iris-setosa':
        axes[2][1].scatter(float(iris[i][0]),float(iris[i][3]),c='blue',marker='s')
    if iris[i][4] == 'Iris-versicolor':
        axes[2][1].scatter(float(iris[i][0]),float(iris[i][3]),c='red',marker='^')
    if iris[i][4] == 'Iris-virginica':
        axes[2][1].scatter(float(iris[i][0]),float(iris[i][3]),c='green',marker='o')

axes[2][0].set_xlabel('sepal width(cm)')
axes[2][0].set_ylabel('petal length(cm)')
#axes[2][0].legend(loc='upper left')
for i in range(rows):
    if iris[i][4] == 'Iris-setosa':
        axes[2][0].scatter(float(iris[i][1]),float(iris[i][2]),c='blue',marker='s')
    if iris[i][4] == 'Iris-versicolor':
        axes[2][0].scatter(float(iris[i][1]),float(iris[i][2]),c='red',marker='^')
    if iris[i][4] == 'Iris-virginica':
        axes[2][0].scatter(float(iris[i][1]),float(iris[i][2]),c='green',marker='o')


plt.show()

