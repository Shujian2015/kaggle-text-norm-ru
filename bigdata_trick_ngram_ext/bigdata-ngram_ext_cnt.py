
# coding: utf-8

# TODO:
# 
# 1. dictionary with counts
# 
# 2. better way to deal with memory

# ####  Settings
lan = 'ru'
ngram = 5
ngram_ext = 5
debug_mode = False


# ####  Load training data and change to lower case
# https://www.kaggle.com/tunguz/ru-baseline-lb-0-9632
import pandas as pd
import numpy as np

train = pd.read_csv('../input/' + lan + '_train.csv', encoding='utf-8')

d = [{} for _ in range(ngram + 1)] # add 1 so d[n] for n-gram

# #### Save n-gram dict from external datasets
import os
import operator

if debug_mode:
    DATA_INPUT_PATH = r'../input/' + lan + '_with_types_test'
else:
    DATA_INPUT_PATH = r'../input/' + lan + '_with_types'
    
files = os.listdir(DATA_INPUT_PATH)
files.sort()

for file in files:
    train_ext = open(os.path.join(DATA_INPUT_PATH, file), encoding='UTF8')
    print(file)
    before = []   
    after = []
    while 1:
        line = train_ext.readline().strip()
        
        # Break the while loop when reaches the end
        if line == '':
            break

        pos = line.find('\t')
        text = line[pos + 1:]
        if text[:3] == '':
            continue
        arr = text.split('\t')
        
        # Add line to tmp if not <eos> 
        if arr[0] != '<eos>':
            before.append(arr[0])
            after.append(arr[1])
            
        # Convert one sentence to dict    
        else:    
            
            # For each dict
            for i in range(2, ngram_ext + 1):
                # Modify 'after'
                for j in range(len(before)):
                    if after[j] == '<self>' or after[j] == 'sil':
                        after[j] = before[j]
                        
                # For tokens
                for j in range(len(before) - i + 1):
                    key = [before[j + k] for k in range(i)]
                    value = [after[j + k] for k in range(i)]
                    
                    if key != value: # Only save modified ones
                        if tuple(key) in d[i]:
                            if tuple(value) in d[i][tuple(key)]:
                                d[i][tuple(key)][tuple(value)] += 1
                            else:
                                d[i][tuple(key)][tuple(value)] = 1
                        else:          
                            d[i][tuple(key)] = {tuple(value):1}            
            
            # Clear arrays　　
            before = []   
            after = []

    train_ext.close()

# #### Save n-gram dict from training set
grouped = train.groupby('sentence_id')

# For each dict
for i in range(2, ngram + 1):
    print('Working on ' + str(i) + '-gram')
    
    # For each sentence
    for name, group in grouped:
        before = group['before'].values.tolist()
        after  = group['after'].values.tolist() 
        
        # For tokens
        for j in range(len(before) - i + 1):
            key = [before[j + k] for k in range(i)]
            value = [after[j + k] for k in range(i)]
            # d[i][tuple(key)] = value
            if tuple(key) in d[i]:
                if tuple(value) in d[i][tuple(key)]:
                    d[i][tuple(key)][tuple(value)] += 1
                else:
                    d[i][tuple(key)][tuple(value)] = 1
            else:          
                d[i][tuple(key)] = {tuple(value):1}              
            
            
# Find the max cnt in dictionary and modify
for i in range(2, ngram + 1):
    for key, v_dict in d[i].items():
        max_key = max(v_dict, key=lambda k: v_dict[k])    
        d[i][key] = list(max_key)
        
            
# ####  Load test data
test = pd.read_csv('../input/' + lan + '_test_2.csv')
test['id'] = test['sentence_id'].astype(str) + '_' + test['token_id'].astype(str)


# #### Load bigdatatrick output
bigdata = pd.read_csv('baseline_ext_en_2.csv')
test['after'] = bigdata['after'].values.tolist()


# #### Cover bigdatatrick output by n-gram results
before = test['before'].values.tolist()
token_id = test['token_id'].values.tolist()
after = test['after'].values.tolist()

for i in range(2, ngram + 1):
    print('Working on ' + str(i) + '-gram')
    
    for j in range(0, len(test) - i + 1): 
        # n-grams to check
        key = [before[j + k] for k in range(i)]
        key = tuple(key)
        
        # Need to check token_id is 0 (tokens should be in on sentence)
        if key in d[i] and (not 0 in token_id[j: j+i]):  
            for k in range(i):
                after[j + k] = d[i][key][k]

    test['after'] = after


# #### Save to output
test[['id','after']].to_csv('submission_' + str(ngram) + '_e' + str(ngram_ext) + 'gram-bigdata_ext_cnt_2.csv', index=False)
print('done')



