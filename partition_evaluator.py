#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 06:26:17 2018

@author: krisjan

this module used for assigning observations to cells, evaluating information
gain, getting the probability score with each cell and prediction
"""
import pandas as pd
import numpy as np
import pbar

#%% information gain calculations

class PartitionEvaluator:

    def __init__(self):
        pass

    def gini_impurity(self, labels):
        label_count = labels.value_counts()
        gi = 1 - label_count.map(
            lambda x: (x/len(labels))**2
        ).sum()
        return gi

    def info_gain(self, df_vic, labels):
        base_impurity = self.gini_impurity(labels)
        ttl_labels = len(labels)
        split_impurity = sum(
            [len(labels[df_vic[i]]) * self.gini_impurity(labels[df_vic[i]]) for i in df_vic.index]
        )/ttl_labels
        return base_impurity - split_impurity

    def get_gain_scores(self, pop, X_train, y_train):
        '''
        This is used to evaluate a population of partitions outside of the class structure
        :param pop:
        :param X_train:
        :param y_train:
        :return:
        '''
        df_scores = pd.Series(index = pop.index)
        for individual in pop.index:
            df_vic = self.get_containers(X_train, pop[individual])
            df_scores[individual] = self.info_gain(df_vic, y_train)
        return df_scores
    #%% get observations/rows of data contained in each cell

    def get_container(self, row, cubes, norm):
        distances = {}
        for i in cubes.columns:
            distances[i] = np.linalg.norm(cubes[i].values-row.values,ord=norm)
        return min(distances, key=distances.get)

    def get_containers(self, df, cubes, norm):
        df_row_cube = {}
    #    feature_len = len(df)
        for i, row in enumerate(df.index):
            df_row_cube[row] = self.get_container(df.loc[row,:], cubes, norm)
    #        if feature_len > 300:
    #            pbar.updt(feature_len,i)
        df_row_cube = pd.Series(df_row_cube)
        df_row_cube = pd.DataFrame(df_row_cube, columns=['cube'])
        df_row_cube['row'] = df_row_cube.index
        df_rows_in_cube = df_row_cube.groupby('cube')['row'].apply(list)
        return df_rows_in_cube
    #%% prediction

    def cube_prob(self, df_vic, labels):
        cube_prob = pd.DataFrame(pd.Series(index = df_vic.index),columns=['nr_labels'])
        for ind in df_vic.index:
            cube_prob.loc[ind,'nr_labels'] = len(labels[df_vic[ind]])
            cube_prob.loc[ind,'probability'] = sum(labels[df_vic[ind]])/cube_prob.loc[ind,'nr_labels']
        return cube_prob

    def get_probs(self, part, X_train, y_train):
        df_vic = self.get_containers(X_train, part)
        df_probs = self.cube_prob(df_vic, y_train)
        return df_probs

    def predict(self, part, df_probs, X_test):
        df_vic = self.get_containers(X_test, part)
        print(df_vic)
        df_predict = pd.Series(index = X_test.index)
        for ind in df_vic.index:
            for row in df_vic[ind]:
                df_predict[row] = df_probs.loc[ind,'probability']
        return df_predict
    
    