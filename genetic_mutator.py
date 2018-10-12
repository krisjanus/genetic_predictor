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
def gen_split(column, lower, upper, probability=1, cur_split=np.nan):
    if probability >= random.random():
        if column.dtypes == 'float':
            split_p = random.uniform(lower, upper)
        if column.dtypes == 'int':
            split_p = random.randint(lower, upper)
        return split_p
    else:
        return cur_split

def mutate(surv_series, bounds, probability = 1):
    for ind in surv_series.index:
        for col in surv_series[ind]:
            centre = surv_series[ind].loc[:,col]
            new_centre = {}
            for index in centre:
                lower = bounds.loc[index,'lower']
                upper = bounds.loc[index,'upper']
                new_centre[index] = gen_split(centre[index], lower, upper, 
                          probability, centre[index])
    
        
def breed_pair():
    return