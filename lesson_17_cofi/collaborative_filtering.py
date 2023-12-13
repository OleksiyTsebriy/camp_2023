''' this is manual implementation of collaborative filtering 190731
Used by http://localhost:8888/notebooks/science/studies/otsebriy/conductor_tools/cats/1895_estimate_msv_for_pmi_keywords/estimate_msv_for_pmi_with_cofi_v_0-Copy1.ipynb
'''

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def J(Y, R, X, Theta, lambd):
    '''
    params: 1d vector  of X and Theta
    :return expression for cost function
    '''
    assert (X.shape[1] == Theta.shape[0])
    h = X @ Theta

    try:
        assert (h.shape == Y.shape)
    except:
        print('h.shape {} !=Y.shape {}'.format(h.shape, Y.shape))

    J = 1 / 2 * np.sum(((h - Y) * R) ** 2) + lambd / 2 * np.sum(X ** 2) + lambd / 2 * np.sum(Theta ** 2)
    try:
        assert (len(J.shape) == 0)
    except:
        print('J is not raw number. J.shape = ', J.shape)

    return J


def J_derivative(Y, R, X, Theta, num_movies, num_users, num_features, lambd):
    cost_matr = (X @ Theta - Y) * R  # n_movies * n_users

    X_grad = cost_matr @ Theta.T
    Theta_grad = (cost_matr.T @ X).T

    try:
        assert (X_grad.shape == X.shape)
        assert (Theta_grad.shape == Theta.shape)
    except:
        print('Check gradient calculus')

    # Regularization part :
    X_grad += lambd * X
    Theta_grad += lambd * Theta

    return X_grad, Theta_grad


def fit(Y, R, num_features=10, alpha=0.0001, lambd=.01, eps=.1, max_iter=1000, step=100, verbose=0):
    num_movies, num_users = Y.shape

    if verbose:
        print('Running gradient descent with alpha= {}, lambda= {}, eps= {}, max_iter= {}'.format(
            alpha, lambd, eps, max_iter))

    #     X= params[:num_movies*num_features].reshape(num_movies,num_features)
    #     Theta = params[num_movies*num_features:].reshape(num_features,num_users)

    np.random.seed(2019)
    X = np.random.randn(num_movies, num_features)
    Theta = np.random.randn(num_features, num_users)

    J_hist = [-1]  # used for keeping J values. Init with -1 to avoid 0 at first iter
    continue_iter = True  # flag to continue next iter (grad desc step)
    iter_number = 0  # used for limit by max_iter

    try:
        while continue_iter:
            # Do step of gradient descent
            X_grad, Theta_grad = J_derivative(Y, R, X, Theta, num_movies, num_users, num_features, lambd)
            X = X - alpha * X_grad
            Theta = Theta - alpha * Theta_grad

            # keep history of J values
            J_hist.append(J(Y, R, X, Theta, lambd))
            # check criteria of exit (finish grad desc)
            if iter_number > max_iter:  # if limit succeeded
                continue_iter = False
                print('iter_number> max_iter')
            elif np.abs(J_hist[iter_number - 1] - J_hist[iter_number]) < eps:  # if accuracy is succeeded
                continue_iter = False
                print('J_hist[iter_number]={}'.format(J_hist[iter_number]))
            iter_number += 1

            if verbose and iter_number % step == 0:
                print('{}: {}'.format(iter_number, J_hist[iter_number - 1]))
        return X, Theta, J_hist
    except Exception as e:
        print('Training is interrupted due to error:', e)
        return X, Theta, J_hist


# def draw_cost_changes(J_hist):
#     J_hist=J_hist[1:]
#     plt.figure()
#     plt.scatter(np.arange(0,len(J_hist)),J_hist,s=20,marker='.',c='b')
#     plt.xlabel('Iterations')
#     plt.ylabel('Cost')
#     plt.show()


def normalize_Y(Y,R, n_0):
    Ymean = np.zeros((n_0, 1))
    Ynorm = np.zeros(Y.shape)
    for i in range(n_0):
        idx = R[i,:] == 1
        Ymean[i] = Y[i, idx].mean()
        Ynorm[i, idx]= Y[i, idx] - Ymean[i]
    return Ymean, Ynorm



def fit_collaborative_filtering(Y, R, n_features=20, max_iter=50000, verbose=1, return_J_hist= False):
    '''
        Y: df of provided values
        R: df of 0 and 1 - marked values as provided (e.g. R is 1 for elements of Y that are not 0)
    '''

    scale = Y.max() - Y.min()
    Y_scaled = Y / scale * 10
    n_0 = Y_scaled.shape[0]
    Ymean, Ynorm = normalize_Y(Y_scaled.values, R.values, n_0)
    X, Theta, J_hist = fit(Ynorm, R.values, num_features=n_features, alpha=0.0005, lambd=1, max_iter=max_iter,
                           eps=.01, step=50, verbose=verbose)
    # if verbose:
    #     draw_cost_changes(J_hist)

    pred = X @ Theta
    pred_rescaled = (pred + Ymean) * scale.values / 10

    df_results_pivot= pd.DataFrame(pred_rescaled , index= Y.index, columns = Y.columns)

    if return_J_hist:
        return df_results_pivot, J_hist
    else:
        return df_results_pivot

# ========== Let's try to cover  more steps here:

def convert_to_matrix(df, index, columns, values):
    '''
        e.g. values='average_msv', index='phrase', columns='locode'
    '''
    df_target=df.pivot_table(index=index, columns=columns, values=values, aggfunc=np.max, dropna= False)
    return df_target

def fit_target(df_target, index, columns, values, n_features=20, max_iter=5000, verbose=1):
    Y= df_target.fillna(0) # not sure it is necessary
    R= df_target.applymap(lambda x: 0 if np.isnan(x) else 1)

    df_results_pivot, J_hist = fit_collaborative_filtering(Y, R, n_features=n_features, max_iter=max_iter, verbose=verbose, return_J_hist= True)

    df_results_pivot_temp= pd.DataFrame(df_results_pivot.to_records())
    df_results = pd.melt(df_results_pivot_temp,
                id_vars=index, # 'iid',
                value_vars=list(df_results_pivot_temp.columns[1:]),
                var_name= columns,# 'uid',
                value_name= '{}_pred'.format(values)) # 'rating_pred')
    return df_results,  J_hist




def round_to_existing(val, existing_unique_values):
    return existing_unique_values[np.argmin(np.abs(existing_unique_values-val))]


def fill_missed(df_input, df_missed= None, compute_average_for_blank_columns= False, n_features=20, max_iter=5000, verbose=1, return_J_hist= False):
    '''
    :param df_input: df with three columns - index, columns, values- all values are provided
    :param df_missed- df with 2 or 3 columns - but first two are index, columns
    :return:  df_predict - that contains all possible combinations
    '''
    index, columns, values= list(df_input)
    print ('index: {}, columns: {}, values: {}'.format(index, columns, values))

    if compute_average_for_blank_columns:
        df_input_atleast= df_input
    else:
        df_input_atleast = df_input.dropna()

    df_target = convert_to_matrix(df_input_atleast, index=index, columns=columns, values=values)
    df_pred, J_hist = fit_target(df_target, index, columns, values, n_features=n_features, max_iter=max_iter, verbose=verbose)

    existing_unique_values = np.array(sorted(df_input[values].unique()))
    df_pred['{}_pred_round'.format(values)]= df_pred['{}_pred'.format(values)].apply(lambda x: round_to_existing(x,existing_unique_values))
    df_pred[index]=df_pred[index].astype(df_input.dtypes[index])
    df_pred[columns] = df_pred[columns].astype(df_input.dtypes[columns])
    df_pred= df_pred.merge(df_input, how='outer', on = ([index, columns]))
    df_pred= df_pred[[index,columns, values, '{}_pred_round'.format(values), '{}_pred'.format(values)]]

    if df_missed is None:
        df_out= df_pred
    else:
        df_out = df_missed.merge(df_pred, how= 'left', on= ([index, columns]))

    if return_J_hist:
        return df_out, J_hist
    else:
        return df_out



if __name__ == '__main__':
    def get_data():
        df_filmtrust = pd.read_csv('/Users/new/science/studies/otsebriy/conductor_tools/cats/1895_estimate_msv_for_pmi_keywords/filmtrust/ratings.txt', sep=' ', names=['uid', 'iid', 'rating'])
        print('len(df_filmtrust)= {:,}'.format(len(df_filmtrust)))
        # df_filmtrust=df_filmtrust[['iid','uid','rating']]
        print(df_filmtrust.head(30))
        return df_filmtrust

    df_filmtrust = get_data()
    df_pred, J_hist= fill_missed (df_filmtrust, compute_average_for_blank_columns= False, n_features=20, max_iter=5000, return_J_hist= True)
    df_pred.to_csv('pred_filmtrust_190801_manual.csv')
    print (df_pred.head(100))