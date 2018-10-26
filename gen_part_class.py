#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 06:21:53 2018

@author: krisjan
"""
import pandas as pd
import partition_evaluator as part_eval

class partition_classifier():
    def __init__(self, df_partition, X_train = None, y_train = None, df_vic = None, 
                 df_proba = None, info_gain = None):
        self.part = df_partition
        self.vectors_in_cubes = df_vic
        self.probs_in_cube = df_proba
        self.info_gain = info_gain
        self.training_data = X_train
        self.training_labels = y_train
        
    def colonize(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.vectors_in_cubes = part_eval.get_containers(X_train, self.part)
        self.part = self.part.loc[:,self.vectors_in_cubes.index].copy()
        self.probs_in_cube = part_eval.cube_prob(self.vectors_in_cubes, y_train)
        self.info_gain = part_eval.info_gain(self.vectors_in_cubes, y_train)
    
    def predict(self, X_test):
        df_vic = part_eval.get_containers(X_test, self.part)
        print(df_vic)
        df_predict = pd.Series(index = X_test.index)
        for ind in df_vic.index:
            for row in df_vic[ind]:
                df_predict[row] = self.probs_in_cube.loc[ind,'probability']
        return df_predict