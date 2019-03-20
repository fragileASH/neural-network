
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# Build the set X and Y for perceptron
def build_set(data,class_name):
    [rows,cols]=data.shape
    x = np.array(data[:,0:4])
    x = x.astype(np.float64)
    y= np.zeros(rows)
    for i in range(len(y)):
        if data[i][4] == class_name:
            y[i] = 1
    return x,y
#Iris-virginica
#Iris-versicolor
#Iris-setosa
#[iris_x,iris_y] = build_set(iris,'Iris-virginica')


# In[3]:


def test_acc(x,y,w,b):
    count = 0
    for i in range(len(x)):
        p = sum(w * x[i]) - b
        if (y[i] == 1) & (p < 0 ):
            count = count + 1
        if (y[i] == 0) & (p >= 0 ):
            count = count + 1
    count = float(count)
    return count


# In[4]:


def perceptron(classname,epoch):
    acc = 0.995
    print('This is a standard perceptron without learning rate')
    print('Perceptron for ',classname)
    iris = np.loadtxt('iris.data',str,delimiter=',')
    [x,y] = build_set(iris,classname)
    np.random.seed(300)
    w = np.random.rand(4)
    #random initial weights set
    print('The initial weights is: ',w)
    count = 0
    b = np.random.rand()
    #random inital bias
    ran = len(x)
    #training times
    for i in range (epoch):
        
        row = np.random.randint(0,ran)
        
        prediction = sum(w*x[row]) - b
        
        if (y[row]==1) & (prediction >= 0):
            w = w - x[row]
            b = b + 1
            
        if (y[row]==0) & (prediction < 0):
            w = w + x[row]
            b = b - 1
        count = count + 1
            
        c = test_acc(x,y,w,b)
        c = c / 150
        if c >= acc:
            break
        # code above is the process of weight changing
    print(count,'times training')
    c = test_acc(x,y,w,b)
    print ('For all data in the iris set, the prediction correct for',c,'times')
    c = c/150
    print('The accuracy is',c)
    print('weight is ',w,' bias is ',b)   


# In[5]:


# perceptron with learninng rate and stopping criterion
def perceptron_lambda(classname,lr,acc,epoch): 
    
    print('This is a perceptron with learning rate')
    print('Perceptron for ',classname)
    count = 0
    iris = np.loadtxt('iris.data',str,delimiter=',')
    [x,y] = build_set(iris,classname)
    np.random.seed(300)
    w = np.random.rand(4)
    #random initial weights set
    print('The initial weights is: ',w)
    b = np.random.rand()
    #random inital bias
    
    #training times
    
    for i in range (epoch):
        
        row = np.random.randint(0,len(x))
        
        prediction = sum(w*x[row]) - b
        
        if (y[row]==1) & (prediction >= 0):
            w = w - lr*x[row]
            b = b + lr
            
        if (y[row]==0) & (prediction < 0):
            w = w + lr*x[row]
            b = b - lr
        c = test_acc(x,y,w,b)
        c = c / 150
        count = count + 1
        if c >= acc: # the stopping condition
            break
        
        # code above is the process of weight changing
    print(count,'times training')
    cn = test_acc(x,y,w,b)
    print ('For all data in the iris set, the prediction correct for',cn,'times')
    cn = cn/150
    print('The accuracy is',c)   
    print('weight is ',w,' bias is ',b) 


# In[6]:


perceptron('Iris-setosa',1000)


# In[7]:


perceptron('Iris-versicolor',10000)


# In[8]:


perceptron('Iris-virginica',600)


# In[9]:


perceptron_lambda('Iris-setosa',0.2,0.99,1000)


# In[10]:


perceptron_lambda('Iris-versicolor',0.05,0.8,10000)


# In[11]:


perceptron_lambda('Iris-virginica',0.03,0.98,10000)

