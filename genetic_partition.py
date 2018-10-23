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
import math
import pbar
import genetic_mutator as gen_mut
import partition_evaluator as part_eval

def get_bounds(column):
    upper = column.max()
    lower = column.min()
    return lower,upper
    
def bounds_df(df):
    bounds=pd.DataFrame(index=df.columns, columns=['lower','upper'])
    for col in df.columns:
        bounds.loc[col,'lower'],bounds.loc[col,'upper'] = get_bounds(df[col])
    return bounds



def train(X_train, y_train, pop_size, gen_size, prob_mutate = .05, 
          mutate_strength = .3, survival_rate = .1, alien_rate = .1):
    bounds = bounds_df(X_train)
    pop = gen_mut.gen_pop(X_train, bounds, pop_size)
    df_scores = part_eval.get_gain_scores(pop, X_train, y_train)
    print('best:', df_scores.sort_values(ascending = False)[:1])
    for i in range(gen_size-1):
        print('generation', i+1)
        pop, nr_surv = gen_mut.new_gen(pop, X_train, df_scores, survival_rate, alien_rate, 
                      pop_size, prob_mutate, mutate_strength, bounds, i+1, keep_originals=False)
        df_scores = pd.concat([df_scores[:nr_surv],part_eval.get_gain_scores(pop[nr_surv:pop_size], X_train, y_train)])
        print('best:', df_scores.sort_values(ascending = False)[:1])
    return pop[list(df_scores.sort_values(ascending = False)[:1].index)].values[0]

       
