#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 06:13:15 2018

@author: krisjan
"""

import os
os.chdir('/Users/krisjan/repos/genetic_predictor')
#%%
import pandas as pd
import numpy as np
import genetic_partition as gen_part
import genetic_mutator as gen_mut
import partition_evaluator as part_eval
#%% developing cube melter/pruning
bounds = X_train.apply(gen_part.get_bounds, axis=0).apply(pd.Series)
bounds.rename(columns={0:'lower',1:'upper'}, inplace=True)
test_ind = gen_mut.gen_cube_centres(X_train, bounds, len(X_train), len(X_train)*2)
vectors_in_cubes = part_eval.get_containers(X_train, test_ind, 1)
# delete cells that contain no data
test_ind = test_ind.loc[:,vectors_in_cubes.index].copy()

new_vic = vectors_in_cubes.copy()
for centroid, df_row_list in vectors_in_cubes.iteritems():
    if len(df_row_list) < 3:
        del test_ind[centroid]
        del new_vic[centroid]
        df_row_cube = {}
        for row in df_row_list:
            df_row_cube[row] = part_eval.get_container(X_train.loc[row,:], test_ind, 1)
        df_row_cube = pd.Series(df_row_cube)
        df_row_cube = pd.DataFrame(df_row_cube, columns=['cube'])
        df_row_cube['row'] = df_row_cube.index
        df_rows_in_cube = df_row_cube.groupby('cube')['row'].apply(list)
        for new_centroid, row_list in df_rows_in_cube.iteritems():
            new_vic[new_centroid] = new_vic[new_centroid] + row_list