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
import gen_part_class as gpt

def get_bounds(column):
    upper = column.max()
    lower = column.min()
    return lower,upper

def delete_empty_cubes(part, df_vic):
    new_part = part.loc[:,df_vic.index].copy()
    return new_part

def train(X_train, y_train, pop_size, gen_size, prob_mutate = .05, 
          mutate_strength = .3, survival_rate = .1, alien_rate = .1,
          min_cubes = 1, max_cubes = 20):
    
    #get bounds of each column
    bounds = X_train.apply(get_bounds, axis=0).apply(pd.Series)
    bounds.rename(index=str, columns={0:'lower',1:'upper'}, inplace=True)
    
    #generate a population of partitions
    pop_parts = gen_mut.gen_pop(X_train, bounds, pop_size, min_cubes, max_cubes)
    
    pop = pop_parts.apply(lambda x: gpt.partition_classifier(x))
    pop.apply(lambda x: x.colonize(X_train, y_train))
    df_scores = pop.apply(lambda x: x.info_gain)
    df_scores.sort_values(ascending = False, inplace=True)
    print('best:', df_scores[:1])
    pop = pop[df_scores.index]
    for i in range(gen_size-1):
        print('generation', i+1)
        pop_parts, nr_surv = gen_mut.new_gen(pop_parts, X_train, df_scores, survival_rate, alien_rate, 
                      pop_size, prob_mutate, mutate_strength, bounds, i+1, min_cubes,
                      max_cubes, keep_originals=False)
        pop_new = pop_parts.apply(lambda x: gpt.partition_classifier(x))
        pop_new.apply(lambda x: x.colonize(X_train, y_train))
        for ind in pop[:nr_surv].index:
            pop_new[ind] = pop[ind]
        df_scores = pop_new.apply(lambda x: x.info_gain)
        df_scores.sort_values(ascending = False, inplace=True)
        print('best:', df_scores[:1])
        pop = pop_new[df_scores.index]
        new_index = ['gen_' + str(i+1) + '_ind_' + str(x) for x in range(len(pop))]
        df_scores.index = new_index
        pop.index = new_index
        pop_parts = pop.apply(lambda x: x.part)
    return pop[list(df_scores[:1].index)].values[0]

       
