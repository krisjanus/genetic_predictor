#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:20:24 2018

@author: krisjan

get a collection of splits
how many splits per variable?
-get some idea of label distribution along variable
labels in a cube?

"""
import pandas as pd
import numpy as np
import random

#%% Evaluation

def gini_impurity(labels):
    label_count = labels.value_counts()
    gi = 1 - label_count.map(lambda x: (x/len(labels))**2).sum()
    return gi

def cube_extractor(cubes, paired_part, index):
    cube = pd.DataFrame(cubes.loc[index,:])
    cube.rename(index=str, columns = {index:'bounds'}, inplace=True)
    for col in cube.index:
        cube.loc[col,'is_upper_bound'] = (cube.loc[col, 'bounds'] == 
                                    paired_part[col][len(paired_part[col])-1])
    return cube

def labels_in_cube(cube, features):
    df = pd.DataFrame(index = features.index)
    for col in cube.index:
        ubt = ((len(cube[col])==3) & 
                (features[col] == cube[col][1]))
        df[col] = ((features[col] >= cube[col][0]) & 
                    ((features[col] < cube[col][1]) | ubt))

    df['in_cube'] = (df.sum(axis=1) == len(df.columns))
    return list(df.loc[df.in_cube,:].index)
    
def info_gain(left_labels, right_labels):
    base_impurity = gini_impurity(pd.concat([left_labels,right_labels],ignore_index=True))
    ttl_labels = len(left_labels) + len(right_labels)
    split_impurity = (gini_impurity(left_labels)*len(left_labels) + 
                        gini_impurity(right_labels)*len(right_labels))/ttl_labels
    return base_impurity - split_impurity
    
#%%

def get_bounds(column):
    upper = column.max()
    lower = column.min()
    return lower,upper
    
def gen_split(column, lower, upper):
    if column.dtypes == 'float':
        split_p = random.uniform(lower, upper)
    if column.dtypes == 'int':
        split_p = random.randint(lower, upper)
    return split_p

def gen_part(df):
    part = pd.Series()
    for col in df.columns:
        lower, upper = get_bounds(df[col])
        split_p = gen_split(df[col], lower, upper)
        part[col] = list(set([lower, split_p, upper]))
        part[col].sort()
    return part

def make_pairs(part):
    part_pairs = pd.Series()
    for col in part.index:
        new_list = []
        for i in range(len(part[col])-1):
            if i != (len(part[col])-2):
                new_list.append(part[col][i:i+2])
            else:
                new_list.append(part[col][i:i+2]+[1])
        part_pairs[col] = new_list
    return part_pairs
        
def get_cubes(paired_part, parent_lists=[[]], index=0):
    if index == len(paired_part):
        return pd.DataFrame([pd.concat(p) for p in parent_lists])
    else:
        child_list = paired_part.iloc[index]
        child_name = paired_part.index[index]
        child_list_labeled = [pd.Series({child_name: c}) for c in child_list]
        return get_cubes(paired_part,[p+[c] for p in parent_lists 
                                         for c in child_list_labeled],index+1)
       
