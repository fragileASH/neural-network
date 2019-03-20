
# coding: utf-8

# In[46]:


import numpy as np
iris = np.loadtxt('iris.data',str,delimiter=',')


# In[47]:


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


# In[48]:


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


# In[49]:


# perceptron with learninng rate and stopping criterion
def perceptron_lambda(classname,lr,acc,epoch): 
    
    #print('This is a perceptron with learning rate')
    #print('Perceptron for ',classname)
    count = 0
    
    [x,y] = build_set(iris,classname)
    np.random.seed(300)
    w = np.random.rand(4)
    #random initial weights set
    #print('The initial weights is: ',w)
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
        if c >= acc:
            break
        
    # The code above is the process of weight changing
    #print(count,'times training')
    cn = test_acc(x,y,w,b)
    #print ('For all data in the iris set, the prediction correct for',cn,'times')
    cn = cn/150
    #print('The accuracy is',c)   
    #print('weight is ',w,' bias is ',b) 
    return w,b


# In[50]:


def label(data):
    [rows,cols]=data.shape
    y= np.zeros(rows)
    for i in range(len(y)):
        
        if data[i][4] == 'Iris-versicolor':
            y[i] = 1
            
        if data[i][4] == 'Iris-setosa':
            y[i] = 2
        
        if data[i][4] == 'Iris-virginica':
            y[i] = 3
            
    return y


# In[51]:


def test_nn_acc(x,y,w1,w2,b1,b2):
    c=0
    confusion_c=[0,0,0]
    a1=np.zeros(len(w1))
    a2=np.zeros(len(w2))
    p = np.zeros((150,3))
    for i in range (len(y)):
        for j in range (len(w1)):
            a1[j] = sum(w1[j] * x[i]) - b1[j]
            if a1[j]<0:
                a1[j]=1
            else:a1[j]=0
        for k in range (len(w2)):
            a2[k] = sum(w2[k] * a1) - b2[k]
            if a2[k] < 0:
                a2[k] = 1
            else:
                a2[k] = 0
            p[i][k] = a2[k]
    for i in range(len(p)):
        if  sum(p[i]) == 1: 
            s = y[i] - 1
            s = int(s)
            if p[i][s] == 1:
                c = c + 1
                confusion_c[s] = confusion_c[s] +1
    c = c / len(y)
    return c,confusion_c


# In[52]:


#versicolor = (1, 0, 0), setosa = (0,1,0), virginica = (0,0,1)


# In[53]:


def neural_network(data,lr,epoch,acc):#(data,learning rate,training times)
    nnc = 0
    lr = float(lr)
    [x,y]=build_set(data,'nmsl')
    y = label(data)
    w11,b11 = perceptron_lambda('Iris-versicolor',0.2,0.99,1000)
    w12,b12=perceptron_lambda('Iris-setosa',0.05,0.8,10000)
    w13,b13=perceptron_lambda('Iris-virginica',0.05,0.98,10000)
    #inital the first layer 
    
    np.random.seed(300)
    w21 = np.random.rand(3)
    w22 = np.random.rand(3)
    w23 = np.random.rand(3)
    b21 = np.random.rand()
    b22 = np.random.rand()
    b23 = np.random.rand()
    #-----------------------------------
    #--------training-------------------
    for i in range(epoch):
        
        n = np.random.randint(0,len(y))
        a11 = sum(w11*x[n]) - b11
        if a11 < 0:
            a11 = 1
        else:a11 = 0
        
        a12 = sum(w12*x[n]) - b12
        if a12 < 0:
            a12 = 1
        else:a12 = 0
        a13 = sum(w13*x[n]) - b13
        if a13 < 0:
            a13 = 1
        else:a13 = 0
        #the first layer
        a1 = np.array([a11,a12,a13]) #the result of the first layer
        a21 = sum(w21 * a1) - b21
        a22 = sum(w22 * a1) - b22
        a23 = sum(w23 * a1) - b23
        #the second layer
      
        if (y[n] == 1) & (a21 >= 0):
            w21 = w21 - lr * a1
            b21 = b21 + lr
        if (y[n] != 1) & (a21 < 0):
            w21 = w21 + lr * a1
            b21 = b21 - lr
            
        if (y[n] == 2) & (a22 >= 0):
            w22 = w22 - lr * a1
            b22 = b22 + lr
        if (y[n] != 2) & (a22 < 0):
            w22 = w22 + lr * a1
            b22 = b22 - lr
            
        if (y[n] == 3) & (a23 >= 0):
            w23 = w23 - lr * a1
            b23 = b23 + lr
        if (y[n] != 3) & (a23 < 0):
            w23 = w23 + lr * a1
            b23 = b23 - lr
        w1=[w11,w12,w13]
        w2=[w21,w22,w23]
        b1=[b11,b12,b13]
        b2=[b21,b22,b23]
        c,con= test_nn_acc(x,y,w1,w2,b1,b2)
        nnc = nnc + 1
        if c >= acc:
            break
        w1=[w11,w12,w13]
        w2=[w21,w22,w23]
        b1=[b11,b12,b13]
        b2=[b21,b22,b23]
        #updating the weights and bias of the second layer
    print('the accuracy is',c)
    print('the training times is',nnc)
    print(con)
    print('confusion matrix')
    print('          ','versicolor','   ','setosa','  ','virginica')
    print('Actual','      ',50,'          ',50,'       ',50)
    print('Prediction','  ',con[0],'          ',con[1],'       ',con[2])
    return w1,w2,b1,b2
    


# In[54]:


[w1,w2,b1,b2] = neural_network(iris,0.3,800,0.98)#(data,learning rate,training times，accuracy)


# In[55]:


def predict(w1,w2,b1,b2):
    x1,x2,x3,x4 = input('please input the parameters of iris，separated by ","').split(',')
    x = np.array([float(x1),float(x2),float(x3),float(x4)])
    a1=np.zeros(len(w1))
    a2=np.zeros(len(w2))
    p = np.zeros((3))
    for j in range (len(w1)):
        a1[j] = sum(w1[j] * x) - b1[j]
        if a1[j]<0:
            a1[j]=1
        else:a1[j]=0
    for k in range (len(w2)):
        a2[k] = sum(w2[k] * a1) - b2[k]
        if a2[k] < 0:
            a2[k] = 1
        else:
            a2[k] = 0
        p[k] = a2[k]
    return p 


# In[56]:


prediction = predict(w1,w2,b1,b2)
print (prediction)


# In[ ]:


#6.3,2.5,5.0,1.9

