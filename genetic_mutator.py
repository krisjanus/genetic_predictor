#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 06:08:52 2018

@author: krisjan
"""

import pandas as pd
import numpy as np
import random
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

def mutate(surv_series, df_dtypes, bounds, probability = 1):
    mut_series = pd.Series()
    for ind in surv_series.index:
        new_part = pd.DataFrame(index = surv_series[ind].index)
        for col in surv_series[ind]:
            centre = surv_series[ind].loc[:,col]
            new_centre = {}
            for index in centre.index:
                lower = bounds.loc[index,'lower']
                upper = bounds.loc[index,'upper']
                col_dtype = df_dtypes[index]
                new_centre[index] = gen_split(col_dtype, lower, upper, 
                          probability, centre[index])
            new_part.loc[:,col] = pd.Series(new_centre)
        new_ind = ind + '_mut'
        mut_series[new_ind] = new_part
    return mut_series
            
    
        
def breed_pair():
    return