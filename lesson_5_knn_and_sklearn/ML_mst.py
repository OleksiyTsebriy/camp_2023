'''
Contains miscelanous tools for ML
'''

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.patches as mpatches
from sklearn import neighbors
import pandas as pd

import matplotlib.pyplot as plt
from scipy import stats as st
import numpy as np
from matplotlib.colors import ListedColormap

def print_df_info(df, nrows = 5,column_values_count =None, describe= False):
    '''prints general information about df'''
    print('Len = {:,}, shape= {}, columns = {},\n{}'.format(len(df), df.shape, list(df), df.head(nrows)))
    if describe:
        print (df.describe())
    if column_values_count:
        print (get_count(df, column=column_values_count))

def print_ndarray_info(data, nrows=5,is_extended= False, label= 'Variable'):
    '''prints general information about df'''
    print('{}: Len = {:,}, shape= {}, \n{}'.format(label, data.shape[0], data.shape, data[:nrows]))
    if is_extended:
        print ('\n{}'.format(st.describe(data)))

def get_count(df, column):
    '''
    sample of using:
        import ML_mst as mst
        mst.get_count(df, column)
    :return: df with only two columns - target and count, sorted by descending
    '''
    return df.groupby(column)[column].agg(['count']).sort_values('count', ascending=False).reset_index() # note without [] for count it will not add new column


def add_count_column(df, column):
    '''
    sample of using:
        import ML_mst as mst
        mst.add_count_column(df, 'fruit_name')
    :return:  df with additional column 'count'
    '''
    df_count= get_count(df, column)
    return pd.merge(df, df_count, how='left', left_on=column,right_on=column)


def plot_data_df(X,y):
    '''
    draws the plot
    :param X: df
    :param y: series
    :return: None
    '''
    y  = y.reshape(1,-1)[0] # make the shape 1 for 12 and since it is ndarray

    positive_indices = (y == 1)
    # print (positive_indices)
    negative_indices = (y == 0)

    print (X[positive_indices][:,0])

    # print (type(positive_indices)) # <class 'pandas.core.series.Series'>
    plt.scatter(X[positive_indices][:,0], X[positive_indices][:,1], s=30, c='black', marker='+',label='positive')
    plt.scatter(X[negative_indices][:,0], X[negative_indices][:,1], s=30, c='green', marker='o', label='negative')
    plt.legend(loc= 2)
    plt.show()


def plot_data_logistic_regression(X,y,legend_loc= 1, title= None):
    '''
    :param X: 2 dimensional ndarray
    :param y:  1 dimensional ndarray. Use y.ravel() if necessary
    :return:
    '''
    positive_indices = (y == 1)
    negative_indices = (y == 0)
    import matplotlib as mpl
    colors_for_points = ['blue', 'green'] # neg/pos
    plt.scatter(X[positive_indices][:,0], X[positive_indices][:,1], s=30, c=colors_for_points [1], marker='+',label='positive')
    plt.scatter(X[negative_indices][:,0], X[negative_indices][:,1], s=20, c=colors_for_points [0], marker='o', label='negative')
    plt.title(title)
    plt.legend(loc= legend_loc)


    plt.show()


def draw_plot_one_variable(X,y, **kwargs):
    import matplotlib.pyplot as plt
    plt.scatter(X,y,s=10,marker='x',c='b')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')

    if 'theta' in kwargs:
        theta = kwargs['theta']
        x= np.linspace(min(X),max(X),100)
        h= theta[0] + theta[1] *x
        plt.plot (x,h,color= 'green')

def draw_cost_function(J_hist,*args,**kwargs):
    '''
    :param J_hist: list of floats
    '''
    J_hist= J_hist[1:]
    import matplotlib.pyplot as plt
    plt.scatter(np.arange(0,len(J_hist)),J_hist,s=20,marker='.',c='b')
    plt.xlabel('Iterations')
    plt.ylabel('Cost function J value')
    parameters = ['{}'.format(arg) for arg in args]

    parameters += ['{}={}'.format(key, value) for key, value in kwargs.items()]

    plt.title('{},\ncompleted iterations ={}'.format(','.join(parameters), len(J_hist)-2)) # len(J_hist)-2) due to first one is -1 (was not iteration), iter + 1  at the end  of the gradient loop
    plt.show()

def run_gradiend_descent(X,y,J,J_derivative, alpha, eps, max_iter = None, verbose= None):
    print ('Running gradient descent with alpha = {}, eps= {}, max_iter= {}'.format(alpha,eps, max_iter))
    theta = np.zeros(shape=(X.shape[1],1))  # init thetas with all 0
    J_hist=[-1] # for history of J values (init with -1 to  avoid o at first iter )
    continue_iter = True # flag to continue next iter (grad desc step)
    iter_number =0 #  for current iter_number  in the loop
    m = X.shape[0]  # number of samples

    while continue_iter:
        # print (theta)
        # Do step of gradient descent
        theta = theta - alpha * 1/ m * J_derivative(theta, X, y) # sending theta as parametr ensures simultaneous update of all thetas
        # keep history of J values
        J_hist.append(J(theta, X, y))
        if verbose:
            print (J_hist[-1])
        # check criteria of exit (finish grad desc)
        if max_iter and iter_number> max_iter: # if max_iter is provided and limit succeeded
            continue_iter = False
        elif np.abs(J_hist[iter_number-1] - J_hist[iter_number])< eps: # if accuracy is succeeded
            continue_iter = False
        iter_number += 1
    return theta, J_hist

def plot_decision_boundary_poly(X, y, degree, clf, title, lambda_reg):
    '''Copied and modified from adspy_shared_utilities.py'''

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)

    # Create color maps
    from matplotlib.colors import  ListedColormap
    import matplotlib as mpl

    colors_for_areas= [mpl.cm.viridis(.45),mpl.cm.viridis(.6)]
    colors_for_points = [mpl.cm.viridis(.3),mpl.cm.viridis(.75)]
    cmap_light = ListedColormap(colors_for_areas)
    cmap_bold  = ListedColormap(colors_for_points)

    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.

    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50

    x1_min, x1_max = X[:, 0].min() - .1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - .1, X[:, 1].max() + 0.1
    # Creates grids of values
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, mesh_step_size),
                         np.arange(x2_min, x2_max, mesh_step_size))

    # numpy.c_  concatenation along the second axis
    # ravel() Returns a contiguous flattened array.
        # x = np.array([[1, 2, 3], [4, 5, 6]])
        # np.ravel(x) = [1 2 3 4 5 6]
    target_samples_grid= (np.c_[xx1.ravel(), xx2.ravel()])# 2-column ndarray # creates the all possible pairs
    target_samples_grid_poly = poly.fit_transform(target_samples_grid)
    Z = clf.predict(target_samples_grid_poly)

    # Put the result into a color plot
    Z = Z.reshape(xx1.shape)
    plt.figure()
    plt.pcolormesh(xx1, xx2, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(X[:, 0], X[:, 1], s=plot_symbol_size, c=y.ravel(), cmap=cmap_bold, edgecolor = 'black')


    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    import matplotlib.patches as mpatches
    patch0 = mpatches.Patch(color=colors_for_points[0], label='negative')
    patch1 = mpatches.Patch(color=colors_for_points[1], label='positive')
    plt.legend(handles=[patch0, patch1])
    plt.title('{}\nLambda = {}'.format(title,lambda_reg))

    plt.xlabel('feature 1')
    plt.ylabel('feature 2')

    plt.show()


def plot_linear_decision_boundary(X,y, theta, title= None):
    '''
    Draw scatter  and line for provided theta
    :param X: 2-dimensional ndarray of shape (m,n)
    :param y: 2-dimensional ndarray of shape (m,1)
    :param theta: 2-dimensional ndarray of shape (n,1)
    :return: None
    '''
    # print ('theta= \n{}'.format(theta))
    # draw the data and boundary line
    positive_indices = (y.ravel() == 1)
    negative_indices = (y.ravel() == 0)
    plt.scatter(X[positive_indices][:, 0], X[positive_indices][:, 1], s=30, c='black', marker='+',
                label='positive')
    plt.scatter(X[negative_indices][:, 0], X[negative_indices][:, 1], s=30, c='green', marker='o',
                label='negative')
    plt.legend()


    x_line = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    y_line = - theta[0,0] / theta[2,0] - theta[1,0] / theta[2,0] * x_line
    plt.plot(x_line, y_line, '-', color='blue')
    plt.title(title)
    plt.show()


def plot_decision_boundary_original_X (X, y, clf, title=None, precision=0.01,plot_symbol_size = 50):

    '''
    Draws the binary decision boundary for X that is nor required additional features and transformation (like polynomial)
    e.g. mst.plot_decision_boundary_original_X (X_scaled, y, clf, title='SVM. C= {}'.format(C)) # just test plotting for future
    Note: make sure to provide the same data as provided for fit (e.g. scaled) but without "one" column
    Copied and modified from adspy_shared_utilities.py'''

    # Create color maps
    from matplotlib.colors import  ListedColormap
    import matplotlib as mpl

    colors_for_areas= [mpl.cm.viridis(.45),mpl.cm.viridis(.6)]
    colors_for_points = [mpl.cm.viridis(.3),mpl.cm.viridis(.75)]
    cmap_light = ListedColormap(colors_for_areas)
    cmap_bold  = ListedColormap(colors_for_points)

    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.

    mesh_step_size = precision #.01  # step size in the mesh

    x1_min, x1_max = X[:, 0].min() - .1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - .1, X[:, 1].max() + 0.1
    # Creates grids of values
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, mesh_step_size),
                         np.arange(x2_min, x2_max, mesh_step_size))

    # numpy.c_  concatenation along the second axis
    # ravel() Returns a contiguous flattened array.
        # x = np.array([[1, 2, 3], [4, 5, 6]])
        # np.ravel(x) = [1 2 3 4 5 6]
    target_samples_grid= (np.c_[xx1.ravel(), xx2.ravel()])# 2-column ndarray # creates the all possible pairs
    # m= target_samples_grid.shape[0]
    # target_samples_grid_1= np.c_[np.ones(shape=(m,1)),target_samples_grid]
    print ('Call prediction for all grid values')
    Z = clf.predict(target_samples_grid)

    # Put the result into a color plot
    Z = Z.reshape(xx1.shape)
    plt.figure()
    plt.pcolormesh(xx1, xx2, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(X[:, 0], X[:, 1], s=plot_symbol_size, c=y.ravel(), cmap=cmap_bold, edgecolor = 'black',alpha=0.75)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    import matplotlib.patches as mpatches
    patch0 = mpatches.Patch(color=colors_for_points[0], label='negative')
    patch1 = mpatches.Patch(color=colors_for_points[1], label='positive')
    plt.legend(handles=[patch0, patch1])
    plt.title(title)

    plt.xlabel('feature 1')
    plt.ylabel('feature 2')

    plt.show()


def plot_multi_class_logistic_regression(X,y,dict_names=None, title =None):
    '''
    Draw the multi class samples of 2 features
    :param X: X 2 ndarray (m,2),
    :param y: vector (m,)
    :param dict_names: dict of values of y and names
    :return: None
    '''
    import matplotlib.pyplot as plt
    # colors = ['r', 'g', 'b', 'y','orange','brown','yellow','grey'] [:len(set(y))]
    colors= ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF']
    # colors = ['green', 'red', 'blue', 'orange','brown','yellow','grey'][:len(set(y))]
    y_unique = list(set(y))
    for i in range (len(y_unique)):
        ind = y == y_unique[i] # vector
        if dict_names:
            plt.scatter(X[ind,0], X[ind,1], c=colors[i], s=30, label=dict_names[y_unique[i]],edgecolor='black', alpha=.7)
        else:
            plt.scatter(X[ind, 0], X[ind, 1], c=colors[i], s=30)
    if title:
        plt.title(title)

    if dict_names:
        plt.legend(frameon=False)

    plt.show()

def plot_decision_boundary(clf, X_train, y_train, X_test=None, y_test= None, title=None, precision=0.01,plot_symbol_size = 50):

    '''
    similar to plot_decision_boundary_original_X  but considers also test samples
    Draws the binary decision boundary for X that is nor required additional features and transformation (like polynomial)
    Note: make sure to provide the same data as provided for fit (e.g. scaled) but without "one" column
    Copied and modified from adspy_shared_utilities.py'''

    # Create color maps
    from matplotlib.colors import  ListedColormap
    import matplotlib as mpl

    colors_for_areas= [mpl.cm.viridis(.45),mpl.cm.viridis(.6)]
    colors_for_points = [mpl.cm.viridis(.3),mpl.cm.viridis(.75)]
    cmap_light = ListedColormap(colors_for_areas)
    cmap_bold  = ListedColormap(colors_for_points)

    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.

    mesh_step_size = precision #.01  # step size in the mesh

    X= np.concatenate([X_train,X_test], axis=0)
    x1_min, x1_max = X[:, 0].min() - .1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - .1, X[:, 1].max() + 0.1
    # Creates grids of values
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, mesh_step_size),
                         np.arange(x2_min, x2_max, mesh_step_size))

    # numpy.c_  concatenation along the second axis
    # ravel() Returns a contiguous flattened array.
        # x = np.array([[1, 2, 3], [4, 5, 6]])
        # np.ravel(x) = [1 2 3 4 5 6]
    target_samples_grid= (np.c_[xx1.ravel(), xx2.ravel()])# 2-column ndarray # creates the all possible pairs
    # m= target_samples_grid.shape[0]
    # target_samples_grid_1= np.c_[np.ones(shape=(m,1)),target_samples_grid]
    print ('Call prediction for all grid values (precision of drawing = {}, you may configure to speed up e.g. precision=0.05)'.format(precision))
    Z = clf.predict(target_samples_grid)

    # Put the result into a color plot
    Z = Z.reshape(xx1.shape)
    plt.figure()
    plt.pcolormesh(xx1, xx2, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(X_train[:, 0], X_train[:, 1], s=plot_symbol_size, c=y_train.ravel(), cmap=cmap_bold, edgecolor = 'black',alpha=0.75)
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='^', s=plot_symbol_size, c=y_test.ravel(), cmap=cmap_bold, edgecolor = 'black',alpha=0.75)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    import matplotlib.patches as mpatches
    patch0 = mpatches.Patch(color=colors_for_points[0], label='negative')
    patch1 = mpatches.Patch(color=colors_for_points[1], label='positive')
    plt.legend(handles=[patch0, patch1])
    plt.title(title)

    plt.xlabel('feature 1')
    plt.ylabel('feature 2')

    plt.show()


def plot_fruit_knn(X, y, n_neighbors):
    if isinstance(X, (pd.DataFrame,)):
        X_mat = X[['height', 'width']].values #as_matrix()
        y_mat = y.values #as_matrix()
    elif isinstance(X, (np.ndarray,)):
        X_mat = X[:, :2]
        y_mat = y
        #   print(X_mat)

    # Create color maps

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#AFAFAF'])

    clf = neighbors.KNeighborsClassifier(n_neighbors, )
    clf.fit(X_mat, y_mat)

    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.

    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    # numpy.c_ Translates slice objects to concatenation along the second axis
    # e.g. np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
    # ravel() Returns a contiguous flattened array.
    # x = np.array([[1, 2, 3], [4, 5, 6]])
    # np.ravel(x) = [1 2 3 4 5 6]

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor = 'black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    patch0 = mpatches.Patch(color='#FF0000', label='apple')
    patch1 = mpatches.Patch(color='#00FF00', label='mandarin')
    patch2 = mpatches.Patch(color='#0000FF', label='orange')
    patch3 = mpatches.Patch(color='#AFAFAF', label='lemon')
    plt.legend(handles=[patch0, patch1, patch2, patch3])


    plt.xlabel('height (cm)')
    plt.ylabel('width (cm)')
    plt.title('KNN k-neighbors = {}'.format(n_neighbors))
    plt.show()




if __name__== "__main__":
    print ('plot_decision_boundary demo')
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_blobs

    X_D2, y_D2 = make_blobs(n_samples=100, n_features=2,
                            centers=8, cluster_std=1.3,
                            random_state=4)
    y_D2 = y_D2 % 2

    X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)
    from sklearn.neural_network import MLPClassifier
    nnclf = MLPClassifier(hidden_layer_sizes=[10], solver='lbfgs',
                          random_state=0).fit(X_train, y_train)

    plot_decision_boundary(nnclf, X_train, y_train,X_test, y_test, precision=0.02)