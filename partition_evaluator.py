#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 06:26:17 2018

@author: krisjan
"""
import pandas as pd
import numpy as np
import pbar

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

def vectors_in_cube(cube, features):
    df = pd.DataFrame(index = features.index)
    for col in cube.index:
        ubt = ((len(cube[col])==3) & 
                (features[col] == cube[col][1]))
        df[col] = ((features[col] >= cube[col][0]) & 
                    ((features[col] < cube[col][1]) | ubt))

    df['in_cube'] = (df.sum(axis=1) == len(df.columns))
    return list(df.loc[df.in_cube,:].index)

def vector_in_cube(cube, row):
    for col in cube.index:
        ubt = ((len(cube[col])==3) & 
                (row[col] == cube[col][1]))
        vic = ((row[col] >= cube[col][0]) & 
                    ((row[col] < cube[col][1]) | ubt))
        if not vic:
            break

    return vic

def vectors_in_cubes_dict(cubes, features):
#    df_vic = {}
#    for row in tqdm(cubes.index):
#        df_vic[row]=vectors_in_cube(cubes.loc[row,:],features)
    df_vic = [vectors_in_cube(cubes.loc[row,:],features) for row in cubes.index]
    return df_vic   

def vector_in_cube_dict(cubes, features):
    df_vic = {}
    feature_len = len(features)
    i=1
    for row in features.index:
        vic = False
        for cube_row in cubes.index:
            vic=vector_in_cube(cubes.loc[cube_row,:],features.loc[row,:])
            if vic:
                df_vic[row] = cube_row
                break
        pbar.updt(feature_len,i)
        i=i+1
    df_vic = pd.Series(df_vic)
    df_vic = pd.DataFrame(df_vic, columns=['cube'])
    df_vic['row'] = df_vic.index
    df_vic = df_vic.groupby('cube')['row'].apply(list)
    return df_vic  
    
def info_gain(df_vic, labels):
    base_impurity = gini_impurity(labels)
    ttl_labels = len(labels)
    split_impurity = sum([len(labels[df_vic[i]])*gini_impurity(labels[df_vic[i]]) 
                            for i in df_vic.index])/ttl_labels
    return base_impurity - split_impurity

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

def get_gain_scores(pop, X_train, y_train):
    df_scores = pd.Series(index = pop.index)
    for individual in pop.index:
        print('Evaluating',individual)
        df_vic = get_containers(X_train, pop[individual])
        df_scores[individual] = info_gain(df_vic, y_train)
    return df_scores

def get_container(row, cubes):
    distances = {}
    for i in cubes.columns:
        distances[i] = np.linalg.norm(cubes[i].values-row.values,ord=1)
    return min(distances, key=distances.get)

def get_containers(df, cubes):
    df_row_cube = {}
    feature_len = len(df)
    i=1
    for row in df.index:
        df_row_cube[row] = get_container(df.loc[row,:], cubes)
        pbar.updt(feature_len,i)
        i=i+1
    df_row_cube = pd.Series(df_row_cube)
    df_row_cube = pd.DataFrame(df_row_cube, columns=['cube'])
    df_row_cube['row'] = df_row_cube.index
    df_rows_in_cube = df_row_cube.groupby('cube')['row'].apply(list)
    return df_rows_in_cube
#%% prediction
    
def cube_prob(df_vic, labels):
    cube_prob = pd.DataFrame(pd.Series(index = df_vic.index),columns=['nr_labels'])
    for ind in df_vic.index:
        cube_prob.loc[ind,'nr_labels'] = len(labels[df_vic[ind]])
        cube_prob.loc[ind,'probability'] = sum(labels[df_vic[ind]])/cube_prob.loc[ind,'nr_labels']
    return cube_prob

def get_probs(part, X_train, y_train):
    df_vic = get_containers(X_train, part)
    df_probs = cube_prob(df_vic, y_train)
    return df_probs
    