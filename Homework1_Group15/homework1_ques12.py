import numpy as np

#### Question 1 ####

#Addition of matrices
def problem_1a (A, B):
	return A + B

#Dot product and addition 
def problem_1b (A, B, C):    
	return np.dot(A,B)-C

#Hardman product and addition
def problem_1c(A, B, C):    
	return A*B+C.T

#Inner product
def problem_1d(x, y):    
	return np.inner(x,y)

#Zeros matrix of size of the given matrix
def problem_1e(A):    
	return np.zeros(A.shape)

#Solving for A-1x
def problem_1f(A,x):
	return np.linalg.solve(A,x)   

#solving for xA-1
def problem_1g(A,x):
	return  np.linalg.solve(A.T,x.T).T 

#Adding the given matrix with an identity matrix*alpha
def problem_1h(A,alpha):
	r,c=A.shape
	return A+np.eye(r)*alpha

#sum of even elements in the given row
def problem_1i (A,i):
    return np.sum(A[i,::2])

#Arithmatic mean when c<=A<=d
def problem_1j(A,c,d):
	return np.mean(A[np.where(np.logical_and(A>=c, A<=d))])

#Eigen vectors corresponding to the k greatest eigen values
def problem_1k(A,k):
	w, v = np.linalg.eig(A)
	idx = np.argsort(w)[::-1] 
	v_k = v[:,idx[:k]]
	return v_k

#multidimensional Gaussian distribution
def problem_1l(x,k,m,s):    
	return np.random.multivariate_normal(x + m, np.eye(x.shape[0])*s,k).T

#randomly permuting the rows of a given matrix
def problem_1m(A):
	P = np.random.permutation(A)
	return(P)

#returning the normalized vector
def problem_1n(x):
	return((x-np.mean(x))/np.std(x))

#repeating the given vector k times
def problem_1o(x,k):
	r,c = np.atleast_2d(x).shape
	if c == 1:
	 return(np.repeat(np.atleast_2d(x),k,axis=1))
	else :
	 return(np.repeat(np.atleast_2d(x),k,axis=0)).T

#Computing pairwise L2 distances of the given x vectors
def problem_1p(X):       
	r,n = np.shape(X)
	C = np.repeat(np.atleast_3d(X),n,axis=2)
	print(C.shape)
	D = np.swapaxes(C,1,2)
	L2_X = np.sqrt(np.sum(np.square(C-D),axis=0))
	return L2_X


#### Question 2 ####

#computes the weigths
def linear_regression (Xtrain,ytrain):  
	Xtrain_t = np.transpose(Xtrain)
	return np.linalg.solve(np.matmul(Xtrain_t,Xtrain),np.matmul(Xtrain_t,ytrain))

#computes the mean square error
def mse(y_pred,y):
	return np.mean(np.square(y_pred-y))/2

#trains and tests the age_regressor ; returns mse of train and test data
def train_age_regressor ():
	# Load data
	X_tr_1 = np.load("age_regression_Xtr.npy")	
	X_te_1 = np.load("age_regression_Xte.npy")

	X_tr = np.reshape(X_tr_1, (-1, 48*48))    
	ytr = np.load("age_regression_ytr.npy")    	
	X_te = np.reshape(X_te_1, (-1, 48*48))    
	yte = np.load("age_regression_yte.npy")

	w = linear_regression(X_tr, ytr)

	y_pred_train = np.matmul(X_tr,w)
	y_pred_test = np.matmul(X_te,w)

	mse_train = mse(y_pred_train,ytr)
	mse_test = mse(y_pred_test,yte)
	
	return mse_train,mse_test

#mse_train,mse_test = train_age_regressor()
# Ans: mse_train = 50.46549593846259; mse_test = 269.1992893033875





