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
best_part = gen_part.train(X_train, y_train, 50, 20, prob_mutate = .05, 
                           mutate_strength = .3, survival_rate = .1, alien_rate = .1)
#%% probabilities associated with each cube in partition
df_probs = part_eval.get_probs(best_part, X_train, y_train)
# get rid of empty cubes
df_vic = part_eval.get_containers(X_train, best_part)
new_best_part = gen_part.delete_empty_cubes(best_part, df_vic)
# try to get a prediction
df_prediction = part_eval.predict(new_best_part, df_probs, X_test)
df_prediction.sort_index(inplace=True)
df_true_test = y_test.sort_index().copy()
#%%
from sklearn.metrics import roc_auc_score, roc_curve
roc_auc_score(df_true_test, df_prediction)
fpr, tpr, thresholds = roc_curve(df_true_test, df_prediction)
plt.plot(fpr,tpr)
#%%
estimator = gpt.partition_classifier(best_part)
estimator.colonize(X_train, y_train)
estimator.info_gain
estimator.predict(X_test)
