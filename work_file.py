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

surv_mut, df_breed_report = gen_mut.mutate(surv_breed, X_train.dtypes, bounds, probability=.01,
                          strength = .2, keep_originals=False)