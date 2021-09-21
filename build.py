#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 08:33:24 2021

@author: peterevans
"""

import pandas as pd
import numpy as np

test = pd.read_csv('./test_cleaned.csv', index_col=0)
train = pd.read_csv('./train_cleaned.csv', index_col = 0)

#%% c) BUILD A CLASSIFIER FOR INCOME 
#% c) investigate and train at least 5 classification methods

#separate data into X and y
y_train = train.label
y_test = test.label
X_train = train.drop(columns = 'label')
X_test = test.drop(columns = 'label')

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#multiple classification models are initialised with their default parameters
names = ['KNN', 'LR', 'DT', 'RF', 'L-SVC', 'NB']
models = [KNeighborsClassifier(), LogisticRegression(max_iter=1000, dual=False),
          DecisionTreeClassifier(), RandomForestClassifier(), 
          LinearSVC(max_iter=1000, dual=False), GaussianNB()]

results = pd.DataFrame(columns = ['Name', 'Training Accuracy', 'Training F-Score',
                                  'Test Accuracy', 'Test F-Score'])
#test each model in turn
for i in range(len(models)):
    cv_results = cross_validate(models[i], X_train, y_train, scoring=('accuracy', 'f1'),
                                cv=5, n_jobs=-1)
    models[i].fit(X_train, y_train)
    y_pred = models[i].predict(X_test)
    
    data_row = {'Name':names[i], 
                'Training Accuracy':np.mean(cv_results['test_accuracy']),
                'Training F-Score':np.mean(cv_results['test_f1']), 
                'Test Accuracy':accuracy_score(y_test, y_pred),
                'Test F-Score':f1_score(y_test, y_pred)}
    
    results = results.append(data_row, ignore_index=True)
    print('DONE:', names[i])
    
#save models
import joblib
for i in range(len(models)):
    joblib.dump(models[i], './models/{}.pkl'.format(names[i]))
    
#%% finetune the hyperparameters of the the models

from sklearn.model_selection import RandomizedSearchCV

#parameters to test on each model
knn_params = {'n_neighbors':[4,5,6,7, 8],
              'leaf_size':[10,20,30,40,50],
              'algorithm':['auto', 'kd_tree', 'ball_tree'],
              'p':[1,2]}
lr_params =  {'penalty': ['l2'],
              'C':[0.001,.009,0.01,.09,1,5,10,25]}
dt_params = {'criterion':['gini', 'entropy'], 
             'splitter':['best', 'random'],
             'max_depth':[2,4,6,8,10,12]}
rf_params = {'n_estimators':[20,40,60,80,100,150,200,250],
             'criterion':['gini', 'entropy'],
             'max_depth': [10, 20, 30, 40, 50, None],
             }
l_svc_params = {'tol':[1e-5, 1e-4, 1e-3, 1e-2,0.1],
                'C':[0.001,.009,0.01,.09,1,5,10,25]}
nb_params = {'var_smoothing':[1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]}


models =[]
models.append(('KNN', KNeighborsClassifier(), knn_params))
models.append(('LR', LogisticRegression(max_iter=1000, dual=False), lr_params))
models.append(('DR', DecisionTreeClassifier(), dt_params))
models.append(('RF', RandomForestClassifier(), rf_params))
models.append(('L-SVC', LinearSVC(max_iter=1000,dual=False), l_svc_params))
models.append(('NB', GaussianNB(), nb_params))

hyper_results = pd.DataFrame(columns=['Name', 'Best Hyperparameters', 'Score'])

#find best hyperparametrs for each of the models 
for name, model, params in models:
    model_cv = RandomizedSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1, n_iter=30)
    model_cv.fit(X_train, y_train)
    
    data_row = {'Name':name,
                'Best Hyperparameters':model_cv.best_params_,
                'Score':model_cv.best_score_}
    
    hyper_results = hyper_results.append(data_row, ignore_index=True)
    print('DONE:', name)
    
#%% recursive feature selection

from sklearn.feature_selection import RFECV

names_for_rfe = ['DT', 'RF', 'L-SVC']
models_for_rfe = [ DecisionTreeClassifier(),
          RandomForestClassifier(), LinearSVC(dual=False,max_iter=1000)]

rfe_results = pd.DataFrame(columns=['Name', 'Number Selected Columns', 'Score',
                                    'Selected Columns'])

#find the best features 
for i in range(len(models_for_rfe)):
    selector = RFECV(models_for_rfe[i], n_jobs=-1, scoring='f1')
    selector.fit(X_train, y_train)
    rankinglist = selector.ranking_.tolist()
    selected_cols = [name for index, name in enumerate(X_train.columns) if rankinglist[index]==1]
    data_row = {'Name':names_for_rfe[i], 'Number Selected Columns':len(selected_cols),
                'Score':np.max(selector.grid_scores_), 'Selected Columns':selected_cols}
    rfe_results = rfe_results.append(data_row, ignore_index =True)
    print('RFE DONE:', names_for_rfe[i])

    
#%% sequential feature selection
from sklearn.feature_selection import SequentialFeatureSelector

names_for_sfs = ['KNN', 'LR', 'DT', 'RF', 'L-SVC', 'NB']
models_for_sfs = [KNeighborsClassifier(), LogisticRegression(max_iter=1000, dual=False),
                     DecisionTreeClassifier(), RandomForestClassifier(), 
                     LinearSVC(max_iter=1000, dual=False), GaussianNB()]
num_features = 10

seq_results = pd.DataFrame(columns=['Name', 'Score', 'Selected Cols'])

for i in range(len(models_for_sfs)):
    sfs = SequentialFeatureSelector(models_for_sfs[i], n_features_to_select=num_features,
                                    direction='forward', scoring='f1', cv=5, n_jobs=-1)
    sfs.fit(X_train, y_train)
    column_bools = pd.Series(sfs.get_support(), index=X_train.columns)
    X_new= X_train.loc[:, column_bools]
    
    cv_results = cross_validate(models_for_sfs[i], X_new, y_train, scoring='f1', cv=5, n_jobs=-1)
        
    data_row = {'Name':names_for_sfs[i],
                'Score':np.mean(cv_results['test_score']),
                'Selected Cols':X_new.columns,
                }
    
    seq_results = seq_results.append(data_row, ignore_index=True)
    print('DONE:', names_for_sfs[i])


#%% combine feature selection and hyperparameters

#initialise models with chosen hyperparameters
names = ['KNN', 'LR', 'DT', 'RF', 'L-SVC', 'NB']
models =[]
models.append(KNeighborsClassifier(**hyper_results['Best Hyperparameters'][0]))
models.append(LogisticRegression(**hyper_results['Best Hyperparameters'][1],
                                        max_iter=1000, dual=False))
models.append(DecisionTreeClassifier(**hyper_results['Best Hyperparameters'][2]))
models.append(RandomForestClassifier(**hyper_results['Best Hyperparameters'][3]))
models.append(LinearSVC(**hyper_results['Best Hyperparameters'][4],
                                  max_iter=1000,dual=False))
models.append(GaussianNB(**hyper_results['Best Hyperparameters'][5]))

combined_results = pd.DataFrame(columns = ['Name', 'Training Accuracy', 'Training F-Score',
                                  'Test Accuracy', 'Test F-Score'])
#test each model in turn with its selected features
for i in range(len(models)):
    X_train_new = X_train[seq_results['Selected Cols'].iloc[i].tolist()]
    X_test_new = X_test[seq_results['Selected Cols'].iloc[i].tolist()]
    cv_results = cross_validate(models[i], X_train_new, y_train, scoring=('accuracy', 'f1'), cv=5, n_jobs=-1)
    models[i].fit(X_train_new, y_train)
    y_pred = models[i].predict(X_test_new)
    
    data_row = {'Name':names[i], 
                'Training Accuracy':np.mean(cv_results['test_accuracy']),
                'Training F-Score':np.mean(cv_results['test_f1']), 
                'Test Accuracy':accuracy_score(y_test, y_pred),
                'Test F-Score':f1_score(y_test, y_pred)}
    
    combined_results = combined_results.append(data_row, ignore_index=True)
    print('DONE:', names[i])


#%% overwrite with the better perfoming models
for i in range(1, len(models)):
    joblib.dump(models[i], './models/{}.pkl'.format(names[i]))
    
#%% investigate ensemble methods (hard voting)
from sklearn.ensemble import VotingClassifier

models =[]
models.append(('KNN', KNeighborsClassifier(**hyper_results['Best Hyperparameters'][0])))
models.append(('LR', LogisticRegression(**hyper_results['Best Hyperparameters'][1], 
                                        max_iter=1000, dual=False)))
models.append(('DR', DecisionTreeClassifier(**hyper_results['Best Hyperparameters'][2])))
models.append(('RF', RandomForestClassifier(**hyper_results['Best Hyperparameters'][3])))
models.append(('L-SVC', LinearSVC(**hyper_results['Best Hyperparameters'][4],
                                  max_iter=1000,dual=False)))
models.append(('NB', GaussianNB(**hyper_results['Best Hyperparameters'][5])))

#create our voting classifier, inputting our models
ensemble = VotingClassifier(models, voting='hard', n_jobs=-1)
#fit model to training data
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
#test our model on the test data
print('VOTING ACCURACY: ', accuracy_score(y_test, y_pred))
print('VOTING F-SCORE: ', f1_score(y_test, y_pred))

joblib.dump(ensemble, './models/{}.pkl'.format('VotingEnsemble'))

#%% investigate ensemble methods (stacking)

from sklearn.ensemble import StackingClassifier
models =[]
models.append(('KNN', KNeighborsClassifier(**hyper_results['Best Hyperparameters'][0])))
models.append(('DR', DecisionTreeClassifier(**hyper_results['Best Hyperparameters'][2])))
models.append(('RF', RandomForestClassifier(**hyper_results['Best Hyperparameters'][3])))
models.append(('L-SVC', LinearSVC(**hyper_results['Best Hyperparameters'][4],
                                  max_iter=1000,dual=False)))
models.append(('NB', GaussianNB(**hyper_results['Best Hyperparameters'][5])))

log_r = LogisticRegression()
stack = StackingClassifier(estimators=models, final_estimator=log_r)
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
print('Stacking ACCURACY: ', accuracy_score(y_test, y_pred))
print('Stacking F-SCORE: ', f1_score(y_test, y_pred))

joblib.dump(stack, './models/{}.pkl'.format('StackingEnsemble'))
