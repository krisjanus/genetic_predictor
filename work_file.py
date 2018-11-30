#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 06:23:15 2018

@author: krisjan

    2. general note: limit min and max cubes to lower number - also to avoid overfitting
"""

import os
#linux 
#dir_name = '/home/krisjan/genetic_predictor'
#mac 
dir_name = '/Users/krisjan/repos/genetic_predictor'
os.chdir(dir_name)
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import genetic_partition as gen_part
import matplotlib.pyplot as plt
import gen_part_class as gpt
from math import floor
import tic_toc
#%% titanic
df = pd.read_csv('data/titanic_prepd.csv')
df = df.set_index('PassengerId')
df.drop(['ticket_numbers'],axis=1,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.drop(['Survived'],axis=1), 
                                                    df['Survived'], 
                                                    test_size=.1)
#%% iris binary
df = pd.read_csv('data/iris_binary_prepd.csv')
#%% bike buyer 
df = pd.read_csv('data/bike_buyer_prepd.csv')
df = df.set_index('ID')
#%%
X_train, X_test, y_train, y_test = train_test_split(df.drop(['label'],axis=1), 
                                                    df['label'], 
                                                    test_size=.1)

#%% test the training module
tic_toc.tic()
size = len(X_train)
# titanic does well with norm 1
# if validation is an integer cross validation is performed and perc_cluster is set to 0
best_part = gen_part.train(df.drop(['Survived'],axis=1), df['Survived'], 500, 10, prob_mutate = .05, 
                           mutate_strength = .3, survival_rate = .1, alien_rate = .1, min_cubes = 20,
                           max_cubes = floor(size/10), min_rows_in_cube = 3, metric = 'acc', validation=5,
                           seed=7, part_norm=1, perc_cluster=.2, jobs=4)
tic_toc.toc()
#%% evaluate best predictor
from sklearn.metrics import roc_curve, accuracy_score
#%%
# first colonize on full train set
best_part.colonize(X_train,y_train)
df_prediction = best_part.predict(X_test)
df_prediction.sort_index(inplace=True)
df_true_test = y_test.sort_index().copy()

best_acc = 0
best_tr = 0
for threshold in np.arange(0,1.01,.01):
    acc = accuracy_score(df_true_test, df_prediction>threshold)
    if acc > best_acc:
        best_acc = acc
        best_tr = threshold   
#%% accuracy score at internal threshold
accuracy_score(df_true_test, df_prediction>best_part.acc_thres)   
#%%
fpr, tpr, thresholds = roc_curve(df_true_test, df_prediction)
plt.plot(fpr,tpr)
plt.grid(True)
plt.ylim(0,1)
plt.xlim(0,1)
plt.axes().set_aspect('equal')
plt.show()


#%% test saving and loading
# ! The best partition should be trained on the full dataset to get the correct
# probabilities per cell. is working with acc you then just need to also find the
# threshold associated with best acc
best_part.colonize(df.drop(['Survived'],axis=1),df['Survived'])
best_part.save('data/titanic_181127.h5')

estimator = gpt.partition_classifier()
estimator.load(filename = 'data/titanic_181130.h5')
df_prediction = estimator.predict(X_test)



#%% Testing parts of modules
import genetic_mutator as gen_mut
import partition_evaluator as part_eval
import pbar
#%%
# generate a population of partitions
pop = gen_part.gen_pop(X_train, 100)
# evaluate each individual and return an information gain score
df_scores = part_eval.get_gain_scores(pop, X_train, y_train)
#%% testing mutation functionality

surv_breed = gen_mut.breed(survivors, df_scores[:5],35)

surv_mut, df_breed_report = gen_mut.mutate(surv_breed, X_train.dtypes, bounds, probability=.05,
                          strength = .2, keep_originals=False)
#%% titanic predict on test
#%% thershold optimisation

best_acc_list = []
best_thr_list = []
for i in range(500):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Survived'],axis=1), 
                                                        df['Survived'], 
                                                        test_size=.2)
    estimator.colonize(X_train, y_train)
    df_prediction = estimator.predict(X_test)
    df_prediction.sort_index(inplace=True)
    df_true_test = y_test.sort_index().copy()
    
    best_acc = 0
    best_tr = 0
    for threshold in np.arange(0,1.01,.01):
        acc = accuracy_score(df_true_test, df_prediction>threshold)
        if acc > best_acc:
            best_acc = acc
            best_tr = threshold 
    
    best_acc_list.append(best_acc)
    best_thr_list.append(best_tr)
    pbar.updt(500,i)
    
#%%
np.mean(best_acc_list)
np.mean(best_thr_list)  
pd.Series(best_thr_list).hist()  
pd.Series(best_thr_list).mode()    
pd.Series(best_acc_list).hist()
#%% train again on full set
estimator.colonize(df.drop(['Survived'],axis=1),df['Survived'])

#%% titanic
df_test = pd.read_csv('data/titanic_test_prepd.csv')
df_test = df_test.set_index('PassengerId')
df_test.drop(['ticket_numbers'],axis=1,inplace=True)
df_probs = estimator.predict(df_test)
df_out = (df_probs > .39).astype(int)
df_out.sum()/len(df_out)
df_out = pd.DataFrame(df_out, columns=['Survived'])
df_out.to_csv('data/titanic_prediction_181130.csv')
