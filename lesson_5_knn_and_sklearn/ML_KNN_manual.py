'''
Contains realization of KNN algorithms without using specialized libraries
'''
import pandas as pd
import ML_mst as mst
import math
import numpy as np


def train_test_split_df(df, percentage= 0.75): # there is alternative from sklearn.model_selection import train_test_split
    '''
    Splits df into two df
    sample of using
        train_df, test_df = train_test_split_df(df,0.80)
    :param percentage: value from 0 to 1. Default value = 0.75 Returns all df as train if  percentage larger than 1
    :return: tuple of train and test df
    '''
    train_amount = int(len(df) * percentage)
    return (df.iloc[:train_amount], df.iloc[train_amount:])


class Scaler():
    def fit_transform(self, X):
        self._min= X.min()
        self._max = X.max()
        return  (X - self._min)/ (self._max- self._min)

    def transform(self, X):
        return (X - self._min) / (self._max - self._min)



class KNN_classifier():
    def __init__(self, k_number= 1):
        '''
        stores the parameter
        :param k_number: amount of nearest neighbours to consider
        '''
        self._k = k_number

    def fit(self, X, y):
        '''
        Just stores whole train set
        :param X: dataframe of features
        :param y: pd.Series of labels
        '''
        self._X = X
        self._y = y

    def compute_distance(self, x, t):
        """
        x, t - pd.Series
        computes distanse from observation x target t
        :return: float
        """
        # print (np.sum((x - t) ** 2))
        return math.sqrt(np.sum((x - t) ** 2))# here are operations with series. Note: make sure to use np.sum to ignore NAN
        # return math.sqrt(sum([(x.loc[index]-t.loc[index]) ** 2 for index in list(x.index.values)]))



    def predict(self, target_observation):
        '''
        computes the label by simple majority of k nearest neighbours
        :param target_observation: dict e.g. target_observation= {'mass': 160,'width': 7,'height':7,'color_score': 0.81}
        :return:
        '''
        df= self._X.copy() # in order to keep X untouched
        df["distance"] = df.apply(self.compute_distance, axis=1, t= target_observation) # calculate distanse to each observation


        y_sr = self._y.copy()
        y_sr.rename('label_reserved_name', inplace= True)
        df = pd.concat([df, y_sr],axis=1)


        df.sort_values("distance", inplace=True)
        df.reset_index(inplace=True, drop=True)
        df_k = df.iloc[0:self._k]  # selected just k samples
        # print (df_k)

        # print (df_k)
        df_grouped = df_k.groupby("label_reserved_name")["label_reserved_name"].\
            agg(["count"]).reset_index().sort_values("count",ascending=False)
        # print (df_grouped)
        return df_grouped.iloc[0]["label_reserved_name"]


    def score(self, X_test, y_test):
        '''
        Computes average correct prediction for provided set
        :param X_test: features
        :param y_test: labels
        :return: float in [0:1]
        '''
        return np.mean(y_test == X_test.apply (self.predict,axis =1))




