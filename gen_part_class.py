#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 06:21:53 2018

@author: krisjan

this file contains the class definition for core partition predictor object
"""
import pandas as pd
import partition_evaluator as part_eval
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

# class defining one partition classifier object
class partition_classifier():
    def __init__(self, df_partition = None, part_norm=1):
        # partition dataframe is a collection of cell centroids
        if df_partition is not None:
            self.part = df_partition.astype('float')
        # which (distance) norm is used to define cell boundaries
        self.part_norm = part_norm
               
    # fit data to the partition and evaluate fitness
    def colonize(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # Series object containing lists of training data indices,
        # indicating which rows of training data are contained in each cell
        self.vectors_in_cubes = part_eval.get_containers(self.X_train, self.part, self.part_norm)
        # delete cells that contain no data
        self.part = self.part.loc[:,self.vectors_in_cubes.index].copy()
        # fraction of positive labels in each cell -  
        # assigns a probability score to each cell
        self.probs_in_cube = part_eval.cube_prob(self.vectors_in_cubes, self.y_train)
        # calculate the information gain for partition
        self.info_gain = part_eval.info_gain(self.vectors_in_cubes, self.y_train)
#        print('\ninformation gain:',self.info_gain)
        # if a test set is present evaluate further metrics, so far only auc
        if (self.X_test is not None):
            df_prediction = self.predict(self.X_test)
            df_prediction.sort_index(inplace=True)
            df_true_test = self.y_test.sort_index().copy()
            self.auc = roc_auc_score(df_true_test, df_prediction)
            self.acc = 0
            self.acc_thres = 0
            for threshold in np.arange(0,1.01,.01):
                acc = accuracy_score(df_true_test, df_prediction>threshold)
                if acc > self.acc:
                    self.acc = acc
                    self.acc_thres = threshold
#            print(' auc:',self.auc,'\nacc:',self.acc)
#        return self

    def prune_cubes(self, min_rows_in_cube):
        new_vic = self.vectors_in_cubes.copy()
        for centroid, df_row_list in self.vectors_in_cubes.iteritems():
            if len(df_row_list) < 3:
                del self.part[centroid]
                del new_vic[centroid]
                df_row_cube = {}
                for row in df_row_list:
                    df_row_cube[row] = part_eval.get_container(self.X_train.loc[row,:], self.part, self.part_norm)
                df_row_cube = pd.Series(df_row_cube)
                df_row_cube = pd.DataFrame(df_row_cube, columns=['cube'])
                df_row_cube['row'] = df_row_cube.index
                df_rows_in_cube = df_row_cube.groupby('cube')['row'].apply(list)
                for new_centroid, row_list in df_rows_in_cube.iteritems():
                    new_vic[new_centroid] = new_vic[new_centroid] + row_list
        self.vectors_in_cubes = new_vic
    
    def predict(self, X_test):
        # assign each data point to a cell
        df_vic = part_eval.get_containers(X_test, self.part, self.part_norm)
        df_predict = pd.Series(index = X_test.index)
        for ind, row_list in df_vic.iteritems():
            for row in row_list:
                # get the probability scores for each data point
                df_predict[row] = self.probs_in_cube.loc[ind,'probability']
        return df_predict
    
    # save - core of predictor is the partition, distance norm and cell probabilities
    # metrics should be optional
    def save(self, filename='gpt_model.h5'):
        self.part.to_hdf(filename, key = 'part', mode='w')
        self.probs_in_cube.to_hdf(filename, key = 'probs_in_cube')
        scalars = pd.Series([self.info_gain],index=['info_gain'],name='scalars')
        if hasattr(self, 'auc'):
            scalars['auc'] = self.auc
        scalars['part_norm'] = self.part_norm
        scalars.astype('float').to_hdf(filename, key='scalars')
    
    def load(self, filename='gpt_model.h5'):
        self.part = pd.read_hdf(filename, key = 'part')
        self.probs_in_cube = pd.read_hdf(filename, key = 'probs_in_cube')
        scalars = pd.read_hdf(filename, key='scalars')
        self.info_gain = scalars['info_gain']
        if 'auc' in list(scalars.index):
            self.auc = scalars['auc']
        self.part_norm = scalars['part_norm']