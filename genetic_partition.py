#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:20:24 2018

@author: krisjan

main training module
"""
import pandas as pd
import genetic_mutator as gen_mut
import gen_part_class as gpt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

def get_bounds(column):
    upper = column.max()
    lower = column.min()
    return lower,upper

def delete_empty_cubes(part, df_vic):
    new_part = part.loc[:,df_vic.index].copy()
    return new_part

def auc_score(individual, X_test, y_test):
    df_prediction = individual.predict(X_test)
    df_prediction.sort_index(inplace=True)
    df_true_test = y_test.sort_index().copy()
    
    return roc_auc_score(df_true_test, df_prediction)

# main function that performs training, and selects best predictor
def train(X_train, y_train, pop_size, gen_size, prob_mutate = .05, 
          mutate_strength = .3, survival_rate = .1, alien_rate = .1,
          min_cubes = 2, max_cubes = 20, metric='info_gain', validation=0, 
          seed=None, part_norm=2):
    
    # define a validation set - best to have this as information gain on its own
    # can be misleading
    if validation == 0:
        X_train = X_train
        y_train = y_train
        X_test = None
        y_test = None
    elif validation < 1:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                    test_size=validation,
                                                    random_state=seed)
    # provision for cross validation
    else:
        skf = StratifiedKFold(n_splits=validation)
        train_test_idx = [(x,y) for x,y in skf.split(X_train, y_train)]
            
    #get bounds of each column
    bounds = X_train.apply(get_bounds, axis=0).apply(pd.Series)
    bounds.rename(index=str, columns={0:'lower',1:'upper'}, inplace=True)
    
    #generate a population of partitions
    pop_parts = gen_mut.gen_pop(X_train, bounds, pop_size, min_cubes, max_cubes)
    # initialize the partition classifier class
    pop = pop_parts.apply(lambda x: gpt.partition_classifier(x, part_norm))
    
    # fit data and evaluate performance for each individual
    print('Colonizing data')
    pop.apply(lambda x: x.colonize(X_train, y_train, X_test, y_test))
    
    if (validation > 0) and (metric == 'auc'):
        df_scores = pop.apply(lambda x: x.auc)
    elif (validation > 0) and (metric == 'acc'):
        df_scores = pop.apply(lambda x: x.acc)
    else:
        df_scores = pop.apply(lambda x: x.info_gain)
        
    df_scores.sort_values(ascending = False, inplace=True)
    print('best:', df_scores[:1])
    pop = pop[df_scores.index]
    for i in range(gen_size-1):
        print('generation', i+1)
        pop_parts, nr_surv = gen_mut.new_gen(pop_parts, X_train, df_scores, survival_rate, alien_rate, 
                      pop_size, prob_mutate, mutate_strength, bounds, i+1, min_cubes,
                      max_cubes, keep_originals=False)
        pop_new = pop_parts.apply(lambda x: gpt.partition_classifier(x, part_norm))
        pop_new.apply(lambda x: x.colonize(X_train, y_train, X_test, y_test))
        for ind in pop[:nr_surv].index:
            pop_new[ind] = pop[ind]
        if (validation>0) and (metric == 'auc'):
            df_scores = pop_new.apply(lambda x: x.auc)
        elif (validation > 0) and (metric == 'acc'):
            df_scores = pop_new.apply(lambda x: x.acc)
        else:
            df_scores = pop_new.apply(lambda x: x.info_gain)
#        df_scores = pop_new.apply(lambda x: x.info_gain)
        df_scores.sort_values(ascending = False, inplace=True)
        print('best:', df_scores[:1])
        pop = pop_new[df_scores.index]
        new_index = ['gen_' + str(i+1) + '_ind_' + str(x) for x in range(len(pop))]
        df_scores.index = new_index
        pop.index = new_index
        pop_parts = pop.apply(lambda x: x.part)
    return pop[list(df_scores[:1].index)].values[0]

       
