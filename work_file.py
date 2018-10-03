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
#%%
df = pd.read_csv('data/titanic_prepd.csv')
df = df.set_index('PassengerId')
df.drop(['ticket_numbers'],axis=1,inplace=True)
#%%
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Survived'],axis=1), 
                                                    df['Survived'], 
                                                    test_size=.2)
#%%
part = gen_part(X_train)
paired_part = make_pairs(part)

cubes = get_cubes(make_pairs(part)) 

test_cube = cube_extractor(cubes, paired_part, 1001)

df_lic = {}
for row in tqdm(cubes.index):
    df_lic[row]=labels_in_cube(cubes.loc[row,:],X_train)
    
gini_impurity(y_train[df_lic[0]])
