#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 06:21:53 2018

@author: krisjan
"""
import pandas as pd
import partition_evaluator as part_eval
from sklearn.metrics import roc_auc_score


class partition_classifier():
    def __init__(self, df_partition, df_vic = None,
                 df_proba = None, info_gain = None, auc = None, part_norm=None):
        self.part = df_partition
#        self.vectors_in_cubes = df_vic
#        self.probs_in_cube = df_proba
#        self.info_gain = info_gain
        self.part_norm = part_norm
#        self.auc = auc
        
        
    def colonize(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.vectors_in_cubes = part_eval.get_containers(self.X_train, self.part, self.part_norm)
        self.part = self.part.loc[:,self.vectors_in_cubes.index].copy()
        self.probs_in_cube = part_eval.cube_prob(self.vectors_in_cubes, self.y_train)
        self.info_gain = part_eval.info_gain(self.vectors_in_cubes, self.y_train)
        print('\ninformation gain:',self.info_gain)
        if (self.X_test is not None):
            df_prediction = self.predict(self.X_test)
            df_prediction.sort_index(inplace=True)
            df_true_test = self.y_test.sort_index().copy()
            self.auc = roc_auc_score(df_true_test, df_prediction)
            print('\nauc:',self.auc)
    
    def predict(self, X_test):
        df_vic = part_eval.get_containers(X_test, self.part, self.part_norm)
        df_predict = pd.Series(index = X_test.index)
        for ind in df_vic.index:
            for row in df_vic[ind]:
                df_predict[row] = self.probs_in_cube.loc[ind,'probability']
        return df_predict