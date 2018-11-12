#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 06:23:15 2018

@author: krisjan
"""

import os
os.chdir('/Users/krisjan/repos/genetic_predictor')
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import genetic_partition as gen_part
import genetic_mutator as gen_mut
import partition_evaluator as part_eval
import matplotlib.pyplot as plt
import gen_part_class as gpt
from math import floor
import tic_toc
#%%
df = pd.read_csv('data/titanic_prepd.csv')
df = df.set_index('PassengerId')
df.drop(['ticket_numbers'],axis=1,inplace=True)
#%%
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Survived'],axis=1), 
                                                    df['Survived'], 
                                                    test_size=.2)

#%% new vector approach
# generate a population of partitions
pop = gen_part.gen_pop(X_train, 100)
# evaluate each individual and return an information gain score
df_scores = part_eval.get_gain_scores(pop, X_train, y_train)
#%% testing mutation functionality

surv_breed = gen_mut.breed(survivors, df_scores[:5],35)

surv_mut, df_breed_report = gen_mut.mutate(surv_breed, X_train.dtypes, bounds, probability=.05,
                          strength = .2, keep_originals=False)
#%% test the training module
tic_toc.tic()
size = len(df*.8)
best_part = gen_part.train(X_train, y_train, 100, 10, prob_mutate = .05, 
                           mutate_strength = .3, survival_rate = .1, 
                           alien_rate = .1, min_cubes = floor(size/2),
                           max_cubes = size, metric = 'auc', validation=.2,
                           seed=7)
tic_toc.toc()
#%% evaluate best predictor
df_prediction = best_part.predict(X_test)
df_prediction.sort_index(inplace=True)
df_true_test = y_test.sort_index().copy()
#%%
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
roc_auc_score(df_true_test, df_prediction)
best_acc = 0
best_tr = 0
for threshold in np.arange(0,1.01,.01):
    acc = accuracy_score(df_true_test, df_prediction>threshold)
    if acc > best_acc:
        best_acc = acc
        best_tr = threshold
#%%
fpr, tpr, thresholds = roc_curve(df_true_test, df_prediction)
plt.plot(fpr,tpr)
plt.grid(True)
plt.ylim(0,1)
plt.xlim(0,1)
plt.axes().set_aspect('equal')
plt.show()
#%%
estimator = gpt.partition_classifier(best_part)
estimator.colonize(X_train, y_train)
estimator.info_gain
estimator.predict(X_test)


