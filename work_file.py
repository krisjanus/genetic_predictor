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
#%%

#%%
parts = gen_part.gen_pop(X_train, 100)
paired_part = part_eval.make_pairs(part)

cubes = part_eval.get_cubes(part_eval.make_pairs(parts[24])) 

df_vic = part_eval.vector_in_cube_dict(cubes, X_train)
info_gain(df_vic, y_train)

ttl_labels = len(y_train)
sum([len(y_train[df_vic[i]])*gini_impurity(y_train[df_vic[i]]) 
                            for i in range(len(df_vic))])/ttl_labels
#%% new vector approach
cubes = gen_part.gen_cube_centres(X_train)
part_eval.info_gain(part_eval.get_containers(X_train, cubes), y_train)
pop = gen_part.gen_pop(X_train, 100)
df_scores = part_eval.get_gain_scores(pop, X_train, y_train)
