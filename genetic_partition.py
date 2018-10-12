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
    
def gen_part(df):
    part = pd.Series()
    for col in df.columns:
        lower, upper = get_bounds(df[col])
        split_p = gen_mut.gen_split(df[col], lower, upper)
        part[col] = list(set([lower, split_p, upper]))
        part[col].sort()
    return part

def gen_centre(df):
    part = pd.Series()
    for col in df.columns:
        lower, upper = get_bounds(df[col])
        split_p = gen_mut.gen_split(df[col], lower, upper)
        part[col] = split_p
    return part

def gen_pop(df, pop_size):
    pop = pd.Series()
    for i in range(pop_size):
        name = 'ind_' + str(i)
        pop[name] = gen_cube_centres(df)
        pbar.updt(pop_size,i+1)
    return pop

def gen_cube_centres(df, cubes=0):
    if cubes == 0:
        cubes = len(df)
    centres = pd.DataFrame(index=df.columns)
    for i in range(cubes):
        name = 'cube_' + str(i)
        centres[name] = gen_centre(df)
        
    return centres

def train(X_train, y_train, pop_size, gen_size, prob_mutate = .2, prob_cross = .6, survival_rate = .1):
    pop = gen_pop(X_train, pop_size)
    df_scores = part_eval.get_gain_scores(pop, X_train, y_train)
    df_scores.sort_values(ascending = False,inplace=True)
    nr_surv = math.ceil(survival_rate*len(df_scores))
    survivors = pop[list(df_scores[:nr_surv].index)]
    mutants = gen_mut.gen
       
