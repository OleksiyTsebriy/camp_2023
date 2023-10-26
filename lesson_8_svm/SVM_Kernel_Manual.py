'''

Extended by Kernels
Uses minimize function from sklearn.optimize
'''
from scipy.io import loadmat
import numpy as np
import ML_mst as mst
from scipy.optimize import minimize
import math
import pandas as pd

class SVM_kernel_manual():
    '''by O.Tsebriy    '''
    # using optimize from scipy

    def __init__(self,C=1, sigma= 1):
        self.C_reg= C
        self.sigma= sigma


    def similarity (self, X):
        # returns for every sample of X the array of similarities to each land_mark from land_marks
        '''
        :param X: ndarray = training set shape= (m,n)
        land_marks:  ndarray (k, n) # usually land_marks = X, so k = m --- stored by fit meth\od
        :return: ndarray shape =  (m,k) - every row is new kernel-features
        '''
            # X = np.random.randint(1, 100, 20).reshape(-1, 2)
            # land_marks = X# np.random.randint(5, 9, 10).reshape(-1, 2)

        land_marks= self.land_marks
        # k,n_1=  land_marks.shape
        # m,n = X.shape
        # print ('X.shape= ',X.shape)
        # print('lm.shape= ', land_marks.shape)
        # if n_1 != n: # check the expected shape
        #     raise SystemExit ('Dimension of landmarks is different from dimension of samples from X: {} but expected {}'.format(n_1,n))
        # Note: It is worth to implement using vectorization...

        # print ('\nmanual result:')
        return np.array([[math.exp(-math.sqrt(np.sum((x - lm) ** 2)) / (2 * self.sigma ** 2)) for lm in land_marks] for x in X])



        # SLOW!!! vectorized solution (used pd.DataFrame) - could not find the way to apply vectorized function that inputs row
        # looks like apply is not vectorized

        # def dist_x(x):
        #     # retruns distance from curret x to all lm.
        #     def dist_lm(lm):
        #         ''' here expected x and lm as data frames with one row each'''
        #         result = math.exp(-math.sqrt(np.sum((x - lm) ** 2)) / (2 * self.sigma ** 2))
        #         return result
        #
        #     df_lm= pd.DataFrame(land_marks)
        #     result_all_lm= df_lm.apply(dist_lm,axis=1)
        #     return result_all_lm

        # df_X= pd.DataFrame(X)
        # result_all_x= df_X.apply(dist_x,axis=1)
        # return  np.array(result_all_x)


    def h(self,x, theta): # hypothesis
        '''
        :param theta:  row np.ndarray -  vector shape= (1,(n+1)) here n - features number
        :param x: row np.ndarray  - vector shape= ((n+1),1) , here n - features number
        :return: expression of h
        '''
        # print(x.shape)
        # print('theta=\n',theta)
        # print ((x @ theta).shape)

        h= np.apply_along_axis((lambda z: 1 if z >= 0 else 0), 1, (x @ theta))

        return h.reshape(-1,1)

    def J(self,theta):
        F_1= self.F_1
        # print ('F_1=',F_1) # make sure the first column is 1 and check nu,ber of features
        y = self.y  # this is expected to be stored as 2-dim  1 col ndarray
        theta = theta.reshape(-1, 1)  # make sure the theta has proper shape since "minimize" func sends this parameter as one dimensional array
        # y = y.reshape(-1, 1) # Unnecessary check just to make sure the shape is as expected
        theta_reg = theta[1:,:]  # exclude interception (i.e. theta_0) for using in regularization
        # print ('F_1 @ theta.shape=', (F_1 @ theta).shape) # just check the multiplization has shape as expected = 2 dim 1 col ndarray

        # cost_1 = np.apply_along_axis((lambda z: float(0 if z >= 1 else -z+1)), 1, (F_1 @ theta))
        # cost_1= cost_1.reshape(-1, 1) # np.apply_along_axis returns 1 dim ndarray but expected 2dim 1 col ndarray
        # # print('cost_1.shape=', cost_1.shape) just check
        #
        # cost_0 = np.apply_along_axis((lambda z: float(0 if z <= -1 else z + 1)), 1, (F_1 @ theta))
        # cost_0= cost_0.reshape(-1,1)# np.apply_along_axis returns 1 dim ndarray but expected 2dim 1 col ndarray

        v_cost_1 = np.vectorize(lambda z: float(0 if z >= 1 else -z + 1))
        cost_1 = v_cost_1(F_1 @ theta)
        v_cost_0 = np.vectorize(lambda z: float(0 if z <= -1 else z + 1))
        cost_0 = v_cost_0(F_1 @ theta)
        J = self.C_reg * np.sum(y * cost_1 + (1-y) * cost_0) + 1/2 *np.sum(theta_reg ** 2)  # cost function
        print ('J=',J) # just to monitor the J function convergence
        return J

    def fit(self, X, y,verbose= False):
        '''
        :param X: ndarray training set - shape (m,n) , m = number of samples, n = number of features
        :param y: ndarray - single column  of lables
        :param verbose: True to display J val changes and draw the plot
        :return: 2dim 1-column ndarray of theta
        '''
        self.land_marks= X   # set all X as landmarks
        print ('start similarity')
        F= self.similarity(X)
        print('complete similarity')

        F_1 = np.c_[np.ones(shape=(F.shape[0], 1)), F] # add column of 1 for theta_0 (intercept)
        self.F_1 = F_1 # store it for using by J  function

        y= y.reshape(-1,1) # make sure the shape is as expected
        self.y = y  # store it for using by J  function

        init_theta= np.ones(shape= (self.F_1.shape[1],1))

        print('start minimize')
        # Note: neither method could resolve more than 100 samples during acceptable time
        solution=  minimize(self.J,init_theta ,method='SLSQP', tol=.1) #,method='SLSQP') #, tol=1e-1, options= {'maxiter': 100})# ,method='SLSQP') method='Nelder-Mead' tol=1e-1  why it does not work
        # this options= {'maxiter':100} also does not work ((
        # BFGS does not work even for 100
        self.theta= solution.x.reshape(-1,1)
        if verbose:
            print(solution) # this displays other results of sol particularly  success: True

    def predict(self, x): # x- ndarray 1*n , n  - number of features
        # print ('By minimize func h() = {}'.format(self.h(x,self.theta)))
        print ('start similarity for {}'.format(x.shape))
        f = self.similarity(x)
        print('complete similarity for {}'.format(x.shape))
        f_1 = np.c_[np.ones(shape=(f.shape[0], 1)), f]  # add column of 1 for theta_0 (intercept)
        m = x.shape[0]

        h_value=  self.h(f_1,self.theta)
        # print ('h_value.shape= {}'.format(h_value.shape))
        print ('start prediction')
        # prediction_value=  np.apply_along_axis(lambda x: int(x>0.5), 1, h_value)
        v_divider = np.vectorize(lambda x: int(x > 0.5))
        prediction_value = v_divider(h_value)
        print('complete prediction')
        # print ('By minimize prediction_value= \n{}'.format(prediction_value))
        return prediction_value



print ('Loading and Visualizing Data ...\n')
mat= loadmat('D:\Python_After_Eleks\Data_Sets\ex6data2_non_linearly_separable.mat')


X= mat['X']# 2dimensional ndarray. shape = (51,2)
y= mat['y'] # 2dimensional ndarray. shape = (51,1)

def cut_array(X,y,M=None): # extract just part of all samples to check
    if M:
        print ('M={}'.format(M))
        arr = np.c_[X,y] # conc in rder to shufflesimultaneously
        from sklearn.utils import shuffle
        # np.random.shuffle(arr) # the same
        arr= shuffle(arr)
        X=arr[:,:2]
        y=arr[:,2]

        X= X[:M]
        y= y[:M]


    return (X,y)


# mst.print_ndarray_info(X)
# mst.print_ndarray_info(y)
M= 2000
(X,y) =  cut_array(X,y,M= M)
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_scaled= scaler.fit_transform(X)

# mst.plot_data_logistic_regression(X,y.ravel(),legend_loc= 2) # to review original data before scaling


if False: #  SVM_kernel_manual - ex6data2_non_linearly_separable.mat
    print('\nTraining Linear SVM ...\n')
    C= 100
    sigma=2
    clf= SVM_kernel_manual(C= C,sigma=sigma)

    clf.fit(X_scaled,y, verbose= True)

    print ('theta= \n{}'.format(clf.theta))

    target_sample = [.8,.9] #expected positive # Predicted value by simple SVM for [0.8, 0.9] = [1]
    tp= np.array(target_sample).reshape(1,-1)
    tp_scaled= scaler.transform(tp)

    predicted_value= clf.predict(tp_scaled)
    print ('Predicted value by simple SVM for {} = {}'.format(target_sample, predicted_value))

    mst.plot_decision_boundary_original_X (X_scaled, y, clf, precision= .05, title='SVM. Gaussian Kernel C= {}, sigma = {}'.format(C,sigma)) # just test plotting for future
else:
    print ('# Manual SVM is skipped. Set True for corresponding section')

# compare with sklearn
print ('Compare with sklearn')
from sklearn.svm import SVC
C= 10000
clf_sklearn= SVC(C=C)
clf_sklearn.fit(X_scaled,y) # note: we pass the X without "1"- columns

mst.plot_decision_boundary_original_X (X_scaled, y, clf_sklearn, title='sklearn Linear SVM. C= {}'.format(C),plot_symbol_size=20)

target_sample = [.8, .9]  # expected positive # Predicted value by simple SVM for [0.8, 0.9] = [1]
tp = np.array(target_sample).reshape(1, -1)
tp_scaled = scaler.transform(tp)
predicted_value= clf_sklearn.predict(tp_scaled)
print ('Predicted value by sklearn SVM for {} = {}'.format(target_sample, predicted_value))


