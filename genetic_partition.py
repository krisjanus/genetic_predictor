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
from multiprocessing import Pool
from functools import partial
import pbar

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

def colonize_ind(ind, X_train, y_train):
    ind.colonize(X_train, y_train)
    return ind

def prune_ind(ind, min_rows_in_cube):
    ind.prune_cubes(min_rows_in_cube)
    return ind

# main function that performs training, and selects best predictor
def train(X_train, y_train, pop_size, gen_size, prob_mutate = .05, 
          mutate_strength = .3, survival_rate = .1, alien_rate = .1,
          min_cubes = 2, max_cubes = 20, min_rows_in_cube = 3, 
          metric='info_gain', validation=0, 
          seed=None, part_norm=2, perc_cluster=0, jobs=None):
    
    #get bounds of each column
    bounds = X_train.apply(get_bounds, axis=0).apply(pd.Series)
    bounds.rename(columns={0:'lower',1:'upper'}, inplace=True)
    
    # define a validation set - best to have this as information gain on its own
    # can be misleading
    x_val_rounds = 1
    if validation == 0:
        train_test_idx = [(X_train.index, None)]
        X_test = None
        y_test = None
        
    elif validation < 1:
        df_train, X_test, train_labels, y_test = train_test_split(X_train, y_train, 
                                                    test_size=validation,
                                                    random_state=seed)
        train_test_idx = [(train_labels.index, y_test.index)]
        
    # provision for cross validation
    else:
        x_val_rounds = validation
        skf = StratifiedKFold(n_splits=validation)
        train_test_idx = [(y_train.iloc[x].index, y_train.iloc[y].index) 
                            for x,y in skf.split(X_train, y_train)]
#        perc_cluster = 0
    
    df_scores = pd.DataFrame()
    #generate a population of partitions
    pop_parts = gen_mut.gen_pop(X_train, bounds, pop_size, min_cubes, max_cubes,
                                perc_cluster = perc_cluster)
    
    for enum, (train_index, test_index) in enumerate(train_test_idx):
        if x_val_rounds > 1: print('Cross validation round',enum+1)
        
        # initialize the partition classifier class
        pop = pop_parts.apply(lambda x: gpt.partition_classifier(x, part_norm))
        
        # fit data and evaluate performance for each individual
        print('Colonizing data')
        X_tr = X_train.loc[train_index,:]
        y_tr = y_train.loc[train_index]
        if test_index is not None:
            X_test = X_train.loc[test_index,:]
            y_test = y_train.loc[test_index]
        
        if jobs is None:
            pop.apply(lambda x: x.colonize(X_tr, y_tr))
            print('Pruning')
            pop.apply(lambda x: x.prune_cubes(min_rows_in_cube))
        else:
            col_part = partial(colonize_ind, X_train=X_tr, y_train=y_tr)
            
#            prune_part = partial(prune_ind, min_rows_in_cube=min_rows_in_cube)
            
            pool = Pool(processes=jobs)

            for i,x in enumerate(pool.imap(col_part, pop)):
                pop.iloc[i]=x
                pbar.updt(len(pop),i)
            print('\nPruning')
#            for i,x in enumerate(pool.imap(prune_part, pop)):
#                pop.iloc[i]=x
#                pbar.updt(len(pop),i)
                
            pool.close()
            pop.apply(lambda x: x.prune_cubes(min_rows_in_cube))
            
        if (validation > 0) and (metric == 'auc'):
            pop.apply(lambda x: x.evaluate(X_test, y_test))
            df_scores[enum] = pop.apply(lambda x: x.auc)
        elif (validation > 0) and (metric == 'acc'):
            pop.apply(lambda x: x.evaluate(X_test, y_test))
            df_scores[enum] = pop.apply(lambda x: x.acc)
        else:
            df_scores[enum] = pop.apply(lambda x: x.info_gain)
        if x_val_rounds > 1: print('Best score:', df_scores[enum].max())
    
    df_scores = df_scores.mean(axis = 1)
    df_scores.sort_values(ascending = False, inplace=True)
    print('best:', df_scores[:1])
    pop = pop[df_scores.index]
    for i in range(gen_size-1):
        print('generation', i+1)
        
        pop_parts, nr_surv = gen_mut.new_gen(pop_parts, X_train, df_scores, survival_rate, alien_rate, 
                          pop_size, prob_mutate, mutate_strength, bounds, i+1, min_cubes,
                          max_cubes, keep_originals=False, perc_cluster = perc_cluster)
        
        df_scores = pd.DataFrame()
        
        for enum, (train_index, test_index) in enumerate(train_test_idx):
            if x_val_rounds > 1: print('Cross validation round',enum+1)
            
            pop_new = pop_parts.apply(lambda x: gpt.partition_classifier(x, part_norm))
            X_tr = X_train.loc[train_index,:]
            y_tr = y_train.loc[train_index]
            if test_index is not None:
                X_test = X_train.loc[test_index,:]
                y_test = y_train.loc[test_index]
                
            if jobs is None:
                pop_new.apply(lambda x: x.colonize(X_tr, y_tr))
                pop_new.apply(lambda x: x.prune_cubes(min_rows_in_cube))
            else:
                col_part = partial(colonize_ind, X_train=X_tr, y_train=y_tr)
                
#                prune_part = partial(prune_ind, min_rows_in_cube=min_rows_in_cube)
                
                pool = Pool(processes=jobs)
    
                for j,x in enumerate(pool.imap(col_part, pop_new)):
                    pop_new.iloc[j]=x
                    pbar.updt(len(pop_new),j)
                
                print('\nPruning')
#                for i,x in enumerate(pool.imap(prune_part, pop_new)):
#                    pop_new.iloc[i]=x
#                    pbar.updt(len(pop_new),i)    
                
                pool.close()
                
                pop_new.apply(lambda x: x.prune_cubes(min_rows_in_cube))
                
            for ind in pop[:nr_surv].index:
                pop_new[ind] = pop[ind]
            if (validation>0) and (metric == 'auc'):
                pop_new.apply(lambda x: x.evaluate(X_test, y_test))
                df_scores[enum] = pop_new.apply(lambda x: x.auc)
            elif (validation > 0) and (metric == 'acc'):
                pop_new.apply(lambda x: x.evaluate(X_test, y_test))
                df_scores[enum] = pop_new.apply(lambda x: x.acc)
            else:
                df_scores[enum] = pop_new.apply(lambda x: x.info_gain)
            if x_val_rounds > 1: print('Best score:', df_scores[enum].max())
        df_scores = df_scores.mean(axis = 1)
        df_scores.sort_values(ascending = False, inplace=True)
        print('best:', df_scores[:1])
        pop = pop_new[df_scores.index]
        new_index = ['gen_' + str(i+1) + '_ind_' + str(x) for x in range(len(pop))]
        df_scores.index = new_index
        pop.index = new_index
        pop_parts = pop.apply(lambda x: x.part)
    return pop[list(df_scores[:1].index)].values[0]

       
