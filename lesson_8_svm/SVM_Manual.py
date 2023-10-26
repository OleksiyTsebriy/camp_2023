'''
Simple SVM without any kernels - just proper cost function
Uses minimize function from sklearn.optimize
'''
from scipy.io import loadmat
import numpy as np
import ML_mst as mst
from scipy.optimize import minimize

class SVM_manual():
    '''by O.Tsebriy    '''
    # using optimize from scipy

    def __init__(self,C=1):
        self.C_reg= C

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
        X= self.X
        # print ('X=',X) # make sure the first column is 1 and check nu,ber of features
        y = self.y  # this is expected to be stored as 2-dim  1 col ndarray
        theta = theta.reshape(-1, 1)  # make sure the theta has proper shape since "minimize" func sends this parameter as one dimensional array
        # y = y.reshape(-1, 1) # Unnecessary check just to make sure the shape is as expected
        theta_reg = theta[1:,:]  # exclude interception (i.e. theta_0) for using in regularization
        # print ('X @ theta.shape=', (X @ theta).shape) # just check the multiplization has shape as expected = 2 dim 1 col ndarray

        # cost_1 = np.apply_along_axis((lambda z: float(0 if z >= 1 else -z+1)), 1, (X @ theta))
        # cost_1= cost_1.reshape(-1, 1) # np.apply_along_axis returns 1 dim ndarray but expected 2dim 1 col ndarray
        v_cost_1 = np.vectorize(lambda z: float(0 if z >= 1 else -z + 1))
        cost_1= v_cost_1(X @ theta)
        # print('cost_1.shape=', cost_1.shape) just check

        # cost_0 = np.apply_along_axis((lambda z: float(0 if z <= -1 else z + 1)), 1, (X @ theta))
        # cost_0= cost_0.reshape(-1,1)# np.apply_along_axis returns 1 dim ndarray but expected 2dim 1 col ndarray

        v_cost_0 = np.vectorize(lambda z: float(0 if z <= -1 else z + 1))
        cost_0 = v_cost_0(X @ theta)

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
        X_1 = np.c_[np.ones(shape=(X.shape[0], 1)), X] # add column of 1 for theta_0 (intercept)
        self.X=X_1 # store it for using by J  function
        m, n = self.X.shape
        y= y.reshape(-1,1) # make sure the shape is as expected
        self.y = y  # store it for using by J  function
        init_theta= np.ones(shape= (n,1))

        solution= minimize(self.J,init_theta ,method='SLSQP', tol=.1) # minimize(self.J,init_theta)# ,method='SLSQP')
        self.theta= solution.x.reshape(-1,1)
        if verbose:
            print(solution) # this displays other results of sol particularly  success: True

    def predict(self, x): # x- ndarray 1*n , n  - number of features
        # print ('By minimize func h() = {}'.format(self.h(x,self.theta)))
        m = x.shape[0]
        x_1= np.c_[np.ones(shape=(m,1)),x]


        h_value=  self.h(x_1,self.theta)
        # prediction_value=  np.apply_along_axis(lambda x: int(x>0.5), 1, h_value)

        v_divider = np.vectorize(lambda x: int(x>0.5))
        prediction_value= v_divider(h_value)

        # print ('By minimize prediction_value= \n{}'.format(prediction_value))
        return prediction_value


if __name__ == '__main__':
    print ('Loading and Visualizing Data ...\n')
    mat= loadmat('D:\Python_After_Eleks\Data_Sets\ex6data1_linearly_separable.mat')


    X= mat['X']# 2dimensional ndarray. shape = (51,2)
    y= mat['y'] # 2dimensional ndarray. shape = (51,1)
    # mst.print_ndarray_info(X)
    # mst.print_ndarray_info(y)

    from sklearn.preprocessing import StandardScaler
    scaler= StandardScaler()
    X_scaled= scaler.fit_transform(X)

    mst.plot_data_logistic_regression(X,y.ravel(),legend_loc= 2) # to review original data before scaling

    print ('\nTraining Linear SVM ...\n')

    C= 100
    clf= SVM_manual(C= C)
    clf.fit(X_scaled,y, verbose= False)

    target_sample = [4,4]
    tp= np.array(target_sample).reshape(1,-1)
    tp_scaled= scaler.transform(tp)


    predicted_value= clf.predict(tp_scaled)
    print ('Predicted value by simple SVM for {} = {}'.format(target_sample, predicted_value))
    print ('theta by simple SVM:\n{}'.format(clf.theta))

    mst.plot_linear_decision_boundary(X_scaled, y, clf.theta, title= 'Simple SVM. C= {}'.format(C))
    mst.plot_decision_boundary_original_X (X_scaled, y, clf, title='SVM. C= {}'.format(C)) # just test plotting for future



    # compare with sklearn
    print ('Compare with sklearn')
    from sklearn.svm import LinearSVC
    clf_sklearn= LinearSVC(C=C)
    clf_sklearn.fit(X_scaled,y) # note: we pass the X without "1"- columns
    print ('clf_sklearn.intercept_= {}'.format(clf_sklearn.intercept_))
    print ('clf_sklearn.coef_= {}'.format(clf_sklearn.coef_))
    theta= np.array([clf_sklearn.intercept_[0],clf_sklearn.coef_[0,0],  clf_sklearn.coef_[0,1]]).reshape(-1,1)

    mst.plot_linear_decision_boundary(X_scaled, y, theta, title= 'sklearn linear SVM. C= {}'.format(C))
    mst.plot_decision_boundary_original_X (X_scaled, y, clf_sklearn, title='sklearn Linear SVM. C= {}'.format(C)) # just test plotting for future

    predicted_value= clf_sklearn.predict(tp_scaled)
    print ('Predicted value by sklearn SVM for {} = {}'.format(target_sample, predicted_value))
    print ('theta by sklearn SVM:\n{}'.format(theta))


    if True: # just demo of cost functions
        x_line = np.linspace(-5, 5, 1000)
        x_line= x_line.reshape(-1,1)

        cost_1 = np.apply_along_axis((lambda z: float(0 if z >= 1 else -z + 1)), 1, (x_line))
        cost_0 = np.apply_along_axis((lambda z: float(0 if z <= -1 else z + 1)), 1, (x_line))

        import matplotlib.pyplot as plt
        plt.scatter(x_line, cost_0, marker = '.', c='green', label='cost_0')
        plt.scatter(x_line, cost_1, marker='.', c= 'blue', label='cost_1')
        plt.legend()
        plt.show()



# Note:
# Without normalizing it provides bad result (not proved)
# Concate 1 columns after scale. Otherwise you will get the first column of X equal to 0 and so the theta_0 will not be calculated
# minimize function calls the objective with 1 dimensional array but not as initial theta
# minimize function returns 1 dimensional array but not as initial theta
# np.apply_along_axis returns 1 dimensional array but not as expected
# matrix multiplication of 2dim  ndarray and 1 dim ndarray is not the same as   2dim  ndarray and 2-dim 1column ndarray
#   So Just make sure you have expected shape by calling shape
# np.apply_along_axis has a bug - it casts to into sometimes
#     e.g. cost_0 = np.apply_along_axis((lambda z: float(0 if z <= -1 else z + 1)), 1, (X @ theta))
# Calling sklearn no need to concat "1" column
#  Cost function could work without normaliation but effect will be always as c= 100
# you may check how the cost function J is converging by printing J before returning value
# Try different methods for optimization if default one finishes with status : failed
