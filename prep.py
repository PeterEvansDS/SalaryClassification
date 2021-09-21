#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#a) IMPORTATION AND HANDLING OF DATA

import pandas as pd
import numpy as np

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'label']
train = pd.read_csv('./adult.data', names=column_names)
test = pd.read_csv('./adult.test', names=column_names)
test.drop(0, inplace=True)

#create joint dataframe
all_data = train.append(test, ignore_index=True)

#convert age to numerical
all_data.age = pd.to_numeric(all_data.age, errors='raise')

#convert numerical data to int type
all_data[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']] \
= all_data[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].apply(
   lambda x: pd.to_numeric(x, downcast='integer'))

#print statistics on numerical columns
print('\n------------- INFO --------------\n')
print(all_data.info())
print('\n------DESCRIPTIVE STATISTICS - NUMERICAL------\n')
print(all_data.describe().round().to_string())

#analyse categorical columns
print(all_data.describe(include=np.object).to_string())
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 
            'relationship', 'race', 'sex', 'native-country', 'label']
print('CAT COLS UNIQUE VALUES: \n')
for i in cat_cols:
    print(i, ' : \n', all_data[i].unique())
    
#standardize label values
label_dict = {' <=50K':'<=50K',  ' >50K':'>50K', ' <=50K.':'<=50K', ' >50K.':'>50K'}
all_data.label = all_data.label.map(lambda x: label_dict[x])

#replace '?' with null
all_data.replace(' ?', np.nan, inplace = True)

#strip unneeded whitespace
for x in cat_cols:
    all_data[x] = all_data[x].str.strip()

#search for duplicate values
duplicated = all_data[all_data.duplicated(keep=False)]
print('\nNUMBER OF DUPLICATE VALUES: ', all_data.duplicated().sum())

#%% b) COMPARE FINDINGS WITH INPUT LABELS

#create age group column
age_bins = [16,20,30,40,50,60, float('inf')]
age_labels = ['16-20', '21-30', '31-40', '41-50', '51-60', '60+']
age_cat = pd.cut(all_data.age, bins = age_bins,
                             labels=age_labels, include_lowest=True)
all_data.insert(1, 'age_cat', age_cat)

#convert columns to numerical for graph
label_binary = {'<=50K':0, '>50K':1}
all_data['label'] = all_data['label'].map(lambda x: label_binary[x])


#%% generate comparison graphs

import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme()
sns.set_palette(sns.color_palette('GnBu', 2))

def stacked_bar (col, rot_angle):
    
    #split group into above and below 50k
    groupby = all_data.groupby(col).count().reset_index()
    groupby2 = all_data.groupby(col).sum().reset_index()
    less_than_50 = groupby.label - groupby2.label
    fraction = groupby2.label/groupby.label
    df1 = pd.DataFrame({col:groupby[col], '>50k':groupby2.label,
                        '<=50k':less_than_50, 'fraction':fraction})
    ax = df1.plot.bar(x=col, stacked=True, rot=rot_angle)
    plt.ylabel('Number of entries')
    
    #add proportion of >50k to bars
    for rect, label in zip(ax.patches,df1['fraction'].round(2)):
        height = rect.get_height()
        ax.text(rect.get_x() + 0.5*rect.get_width(), height, label,
               ha = 'center', va='bottom', size='medium', color='darkblue')
    ax.legend(['>50k', "<=50k"])
    plt.ylabel('Number of entries')
    plt.tight_layout()
    plt.show()
    
stacked_bar('age_cat',0)
stacked_bar('sex',0)
stacked_bar('occupation',45)

#%%prep data for models

#handle categorical data 
all_data.drop(columns = ['age_cat', 'education'], inplace=True)

#binary encoding
sex_binary = {'Female':1, 'Male':0}
all_data['sex'] = all_data['sex'].map(lambda x: sex_binary[x])

#dummy variable encoding
encode_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
all_data = pd.get_dummies(data = all_data, columns=encode_cols, drop_first=True)

#split data back into test and training variables
train = all_data.iloc[:32561]
test = all_data.iloc[32561:]

#save cleaned datasets
train.to_csv('./train_cleaned.csv')
test.to_csv('./test_cleaned.csv')