#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 06:08:52 2018

@author: krisjan

evolutionary module: generates population and applies genetic algorithm
"""

import pandas as pd
import numpy as np
import random
import pbar
from math import floor, ceil
from sklearn.cluster import KMeans
#%%
def gen_part(df, bounds):
    part = pd.Series()
    for col in df.columns:
        lower = bounds.loc[col,'lower'].values
        upper = bounds.loc[col,'upper'].values
        split_p = gen_split(df[col], lower, upper)
        part[col] = list(set([lower, split_p, upper]))
        part[col].sort()
    return part

def gen_centre(df, bounds, probability=1, cur_split=np.nan):
    part = pd.Series()
    for col in df.columns:
        lower = bounds.loc[col,'lower']
        upper = bounds.loc[col,'upper']
        col_dtype = df[col].dtypes
        split_p = gen_split(col_dtype, lower, upper, probability, cur_split)
        part[col] = split_p
    return part

def gen_cluster_centres(df):
    cubes = random.randint(2,floor(len(df)/3))
    centres = pd.DataFrame(KMeans(n_clusters = cubes, init='random').fit(df).cluster_centers_.T, 
                           index=df.columns)
    centres.columns = pd.Series(centres.columns).apply(lambda x:'cube_' + str(x))      
    return centres    

def gen_pop(X_train, bounds, pop_size, min_cubes, max_cubes, prefix='ind'):
    pop = pd.Series()
    print('generating individuals')
    for i in range(pop_size):
        name = prefix + str(i)
        if pop_size % 2 == 0:
            pop[name] = gen_cube_centres(X_train, bounds, min_cubes, max_cubes)
        else:
            pop[name] = gen_cluster_centres(X_train)
        pbar.updt(pop_size,i+1)
    return pop

def gen_cube_centres(df, bounds, min_cubes, max_cubes):
    cubes = random.randint(min_cubes,max_cubes)
    centres = pd.DataFrame(index=df.columns)
    for i in range(cubes):
        name = 'cube_' + str(i)
        centres[name] = gen_centre(df, bounds)       
    return centres

def gen_split(column_dtype, lower, upper, probability=1, cur_split=np.nan):
    if probability >= random.random():
        if column_dtype == 'float':
            split_p = random.uniform(lower, upper)
        if column_dtype == 'int':
            split_p = random.randint(lower, upper)
        return split_p
    else:
        return cur_split

def mutate(surv_series, df_dtypes, bounds, probability = .01, strength = .2, keep_originals=False):
    mut_series = pd.Series()
    df_report = pd.DataFrame()
    for ind in surv_series.index:
        mutant = pd.DataFrame(index = surv_series[ind].index, columns = surv_series[ind].columns)
        
        for col in surv_series[ind].index:
            not_same_sum = 0
            if probability >= random.random():
                axis = surv_series[ind].loc[col,:]
                new_col = {}
                for ind_2 in axis.index:
                    lower = bounds.loc[col,'lower']
                    upper = bounds.loc[col,'upper']
                    col_dtype = df_dtypes[col]
                    new_col[ind_2] = gen_split(col_dtype, lower, upper, 
                              strength, axis[ind_2])
                new_col = pd.Series(new_col)
            else:
                new_col = surv_series[ind].loc[col,:]
            mutant.loc[col,:] = new_col
            not_same_sum = not_same_sum + int(sum(mutant.loc[col,:] != surv_series[ind].loc[col,:]))
            if not_same_sum > 0:
                df_report.loc[ind,'variable'] = col
                df_report.loc[(df_report['variable'] == col)&(df_report.index==ind),'change_count'] = not_same_sum
        if keep_originals and (not_same_sum > 0):
            new_ind = ind + '_mut'
            mut_series[new_ind] = mutant
        elif not keep_originals:
            new_ind = ind
            mut_series[new_ind] = mutant
    return mut_series, df_report
            
              
def breed(surv_series, df_scores, nr_children_limit):
    breed_series = pd.Series()
    list_of_pairs = [(surv_series.index[p1], surv_series.index[p2]) 
                                            for p1 in range(len(surv_series)) 
                                            for p2 in range(p1+1,len(surv_series))]
    list_of_probs =  pd.Series([(df_scores[p1]+df_scores[p2])/2 
                                            for p1 in range(len(df_scores)) 
                                            for p2 in range(p1+1,len(df_scores))])
    list_of_probs.sort_values(ascending=False, inplace=True)
    breed_pair_index = list(list_of_probs.index)
    breed_limit = min([len(breed_pair_index)*2,nr_children_limit])
    for i in breed_pair_index[:breed_limit]:
        index_1, index_2 = list_of_pairs[i]
        col_len = len(surv_series[index_1].index)
        split_point = random.choice(range(col_len))
        new_ind_1 = pd.concat([surv_series[index_1].iloc[:split_point,:], 
                               surv_series[index_2].iloc[split_point:col_len,:]])
        new_ind_2 = pd.concat([surv_series[index_2].iloc[:split_point,:], 
                               surv_series[index_1].iloc[split_point:col_len,:]])
        name_1 = index_1 + '_' + index_2
        name_2 = index_2 + '_' + index_1
        breed_series[name_1] = new_ind_1
        breed_series[name_2] = new_ind_2
    return breed_series

# is this better: initial test can't really see the difference
def new_breed(surv_series, df_scores, nr_children_limit):
    breed_series = pd.Series()
    list_of_pairs = [(surv_series.index[p1], surv_series.index[p2], (df_scores[p1]+df_scores[p2])/2) 
                                            for p1 in range(len(surv_series)) 
                                            for p2 in range(p1+1,len(surv_series))]
    sorted(list_of_pairs, reverse=True, key=lambda x:x[2])
    breed_limit = min([len(list_of_pairs)*2,nr_children_limit])
    for index_1, index_2, breed_prob in list_of_pairs[:breed_limit]:
        print(breed_prob)
        col_len = len(surv_series[index_1].index)
        split_point = random.choice(range(col_len))
        new_ind_1 = pd.concat([surv_series[index_1].iloc[:split_point,:], 
                               surv_series[index_2].iloc[split_point:col_len,:]])
        new_ind_2 = pd.concat([surv_series[index_2].iloc[:split_point,:], 
                               surv_series[index_1].iloc[split_point:col_len,:]])
        name_1 = index_1 + '_' + index_2
        name_2 = index_2 + '_' + index_1
        breed_series[name_1] = new_ind_1
        breed_series[name_2] = new_ind_2
    return breed_series

def new_gen(population, df_train, df_scores, survival_rate, alien_rate, pop_size, prob_mutate, 
          mutate_strength, bounds, iter_nr, min_cubes, max_cubes, keep_originals):
    df_scores.sort_values(ascending = False,inplace=True)
    nr_surv = ceil(survival_rate * pop_size)
    survivors = population[list(df_scores[:nr_surv].index)]
    nr_aliens = floor(alien_rate * pop_size)
    surv_breed = breed(survivors, df_scores[:nr_surv],pop_size-nr_surv-nr_aliens)
    surv_mut, df_report = mutate(surv_breed, df_train.dtypes, bounds, probability=prob_mutate,
                          strength = mutate_strength, keep_originals=keep_originals)
    alien_name = 'gen_'+str(iter_nr)+'_ind_'
    aliens = gen_pop(df_train, bounds, nr_aliens, min_cubes, max_cubes, alien_name)
    df_new_gen = pd.concat([surv_mut, aliens])
    print(df_report)
    return df_new_gen, nr_surv    
    