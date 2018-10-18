#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 06:08:52 2018

@author: krisjan
"""

import pandas as pd
import numpy as np
import random
from math import floor
#%%
def gen_split(column_dtype, lower, upper, probability=1, cur_split=np.nan):
    if probability >= random.random():
        if column_dtype == 'float':
            split_p = random.uniform(lower, upper)
        if column_dtype == 'int':
            split_p = random.randint(lower, upper)
        return split_p
    else:
        return cur_split

#TODO do not select a centre, select a column/axis and mutate along the axis
# possibly with varying probabilities
def mutate(surv_series, df_dtypes, bounds, centre_probability = .01, probability = .01):
    mut_series = pd.Series()
    for ind in surv_series.index:
        new_part = pd.DataFrame(index = surv_series[ind].index)
        not_same_sum = 0
        for col in surv_series[ind]:
            centre = surv_series[ind].loc[:,col]
            if centre_probability >= random.random():
                new_centre = {}
                for index in centre.index:
                    lower = bounds.loc[index,'lower']
                    upper = bounds.loc[index,'upper']
                    col_dtype = df_dtypes[index]
                    new_centre[index] = gen_split(col_dtype, lower, upper, 
                              probability, centre[index])
                new_centre = pd.Series(new_centre)
            else:
                new_centre = centre
            new_part.loc[:,col] = new_centre
            not_same_sum = not_same_sum + int(sum(new_part.loc[:,col] != centre) > 0)
        if not_same_sum > 0:
            new_ind = ind + '_mut'
            mut_series[new_ind] = new_part
    return mut_series
            
              
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