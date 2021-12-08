#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time as t



start = t.time()

def SGD(X,Y,Y_hat,alpha,W):
    
    gradient_w = -np.dot(X.T,(Y - Y_hat))/X.shape[0] + (alpha/X.shape[0])*W
    #print(gradient_w)
    gradient_b = np.mean(Y_hat - Y)
    
    return gradient_w, gradient_b

def softmax(X,W,B):
    
    #print("hello")
    exp = np.dot(X,W) + B
    y_hat = np.zeros(exp.shape)
    for i in range(exp.shape[0]):
        y_hat[i] = np.exp(exp[i])/np.sum(np.exp(exp[i]))
    #print(soft_max.shape)
    
    return y_hat

def cross_entropy(X,Y,W,B,Alpha):
    
    fce = 0
    reg = 0
    #print("X=",X.shape,"Y=",Y.shape)
    
    for i in range(X.shape[0]):
        #print(i)
        #print("x= :",X[i].shape)
        #Y[i] = np.expand_dims(Y[i], axis = 0)
        #print(Y[i].shape)
        Y_pred = softmax(X[i], W, B)
        #print("y_p:",Y_pred.shape,"y:",(Y[i]).shape)
        fce += np.dot(Y[i].T,np.log(Y_pred.T))
    
    for j in range(Y_pred.shape[1]):
        #print(W[j].shape)
        reg += np.dot(W[j].T,W[j])
    
    reg = (Alpha/X.shape[0])*reg
    
    fce = fce*(-1/X.shape[0]) + reg
    
    #print(fce.shape)
    return fce

#def accuracy(Y_pred,Y):
    
#    test_accuracy = np.sum([np.argmax(Y_pred[i])==np.argmax(Y[i]) for i in range(Y.shape[0])])/(Y.shape[0])
    
#    return test_accuracy

best_fce = 10000000
best_epoch = 0
best_batch = 0
best_lr = 0
best_alpha = 0
best_w = 0
best_b = 0

X_train = np.load("fashion_mnist_train_images.npy")
Y_train = np.load("fashion_mnist_train_labels.npy")



X_train = X_train.reshape(X_train.shape[0],-1)/255.0
#Y = Y.reshape(-1,1)
y_train = np.zeros((Y_train.shape[0],10))

for i in range(Y_train.shape[0]):
    for j in range(10):
        #print(j)
        if j == Y_train[i]:
            y_train[i][j] = 1
        else:
            y_train[i][j] = 0

X_test = np.load("fashion_mnist_test_images.npy")
Y_test = np.load("fashion_mnist_test_labels.npy")

y_test = np.zeros((Y_test.shape[0],10))

for i in range(Y_test.shape[0]):
    for j in range(10):
        #print(j)
        if j == Y_test[i]:
            y_test[i][j] = 1
        else:
            y_test[i][j] = 0
X_test = X_test.reshape(X_test.shape[0],-1)/255.0
#Y_test = Y.reshape(-1,1)


split = 0.8*X_train.shape[0]
split = int(split)
np.random.seed(1)
indices = np.random.permutation(X_train.shape[0])
#print(indices)

training_idx = indices[:split]
val_idx = indices[split:]

temp = y_train

x_train = X_train[training_idx,:]
y_train = temp[training_idx,:]
x_val = X_train[val_idx,:]
y_val = temp[val_idx,:]

#print(x_train.shape,y_train.shape,x_val.shape,y_val.shape)

Learning_rate = [0.2, 0.25, 0.15, 0.3]
Batch_size = [128,256,512,1024]
Epochs = [50,100,200,400]
Reg_con = [0.001, 0.0006, 0.0005, 0.05]

#w = np.random.randn(X_train.shape[1],1) #weight matrix
#b = np.random.randn(1)
#print(b.shape)
#print(w.shape)

count = 0



for epsilon in Learning_rate:
    print(epsilon)
    for epoch in Epochs:
        for batch in Batch_size:
            for alpha in Reg_con:
                
                w = np.random.randn(x_train.shape[1],10)/(np.sqrt(x_train.shape[1])) #weight matrix
                b = np.random.randn(1,10)
                
                #print(epochs)
                
                for i in range(epoch):
                    #print("Epsilon:",epsilon,"Epoch:",epoch,"Batch Size:",batch,"Alpha:",alpha)
                    count += 1
                      
                    index = np.random.choice(np.arange(x_train.shape[0]), size = batch, replace = False )
                    X_mini = x_train[index,:]
                    Y_mini = y_train[index,:]
                    #print(X_mini.shape,w.shape,b.shape,Y_mini.shape)
                    Y_hat = softmax(X_mini,w,b)
                    #print(Y_hat.shape)
                        
                    gradient_w, gradient_b = SGD(X_mini, Y_mini, Y_hat, alpha, w)

                        
                    w = w - epsilon*gradient_w
                    b = b - epsilon*gradient_b
                
                    training_cost = cross_entropy(X_mini, Y_mini, w,b, alpha)
                
                validation_cost = cross_entropy(x_val,y_val,w,b,alpha)
                
                print("CE Val:",validation_cost)
                
                if validation_cost < best_fce:
                    best_epoch = epoch
                    
                    best_lr = epsilon
                    best_batch = batch
                    best_alpha = alpha
                    best_w = w
                    best_b = b
                    best_fce = validation_cost
                    
#print("Best Epochs:",best_epoch,"\nBest Learning Rate:",best_lr,"\nBest Batch Size:",best_batch,"\nBest Regularization Rate:",best_alpha)
testing_cost = cross_entropy(X_test, y_test, best_w, best_b, 0)

print("Best Validaation Loss:",best_fce)

print("\nTesting_FCE_LOSS:",testing_cost)

y_hat_test = np.argmax(softmax(X_test,best_w,best_b), axis = 1)

y_test_ = np.argmax(y_test,axis = 1)

acc = np.sum(y_hat_test == y_test_)/len(y_test_)
print("Accuracy:", acc*100)

end = t.time()

print("Time:",((end - start)/60))

