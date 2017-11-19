#coding: utf-8
'''
Modified from:
https://www.kaggle.com/tunguz/ru-baseline-lb-0-9632
https://www.kaggle.com/arccosmos/ru-baseline-lb-0-9799-from-en-thread
'''

import pandas as pd
import numpy as np
import os
import operator
import pickle



#  Settings
lan = 'ru'
ngram = 6          # Ngram for training set
ngram_ext = 6      # External ngram
debug_mode = False # Debug mode just uses one external file
file_to_cover = 'baseline_ext_en_2.csv' # Baseline file
file_output = 'sub_train' + str(ngram) + '_ext' + str(ngram_ext) + 'gram_new'

# External datasets
if debug_mode:
    DATA_INPUT_PATH = r'../input/' + lan + '_with_types_test'
else:
    DATA_INPUT_PATH = r'../input/' + lan + '_with_types'



def input_files(lan, file_to_cover, DATA_INPUT_PATH):
    # Load test data
    test = pd.read_csv('../input/' + lan + '_test_2.csv')
    test['id'] = test['sentence_id'].astype(str) + '_' + test['token_id'].astype(str)

    # Load bigdatatrick output
    bigdata = pd.read_csv(file_to_cover)
    test['after'] = bigdata['after'].values.tolist()

    # Load training
    train = pd.read_csv('../input/' + lan + '_train.csv', encoding='utf-8')

    # Ext
    files = os.listdir(DATA_INPUT_PATH)
    files.sort()
    
    return test, train, files


def ngram_dict_ext(f, n, DATA_INPUT_PATH):    
    # The dictionary for n-gram
    d = {}
    
    #### Ext data ####
    # Check whether need to use ext data
    if (ngram_ext >= n):
        # Loop each file
        for file in [f]:
            train_ext = open(os.path.join(DATA_INPUT_PATH, file), encoding='UTF8')
            print("   " + file)
            before = []   
            after = []
            while 1:
                line = train_ext.readline().strip()

                # Break the while loop when reaches the end
                if line == '':
                    break

                pos = line.find('\t')
                text = line[pos + 1:]
                #if text[:3] == '':
                #     continue
                arr = text.split('\t')

                # Add line to tmp if not <eos> 
                if arr[0] != '<eos>':
                    before.append(arr[0])
                    after.append(arr[1])

                # Convert one sentence to dict    
                else:    
                    # Modify 'after'
                    for j in range(len(before)):
                        if after[j] == '<self>' or after[j] == 'sil':
                            after[j] = before[j]

                    # For tokens
                    for j in range(len(before) - n + 1):
                        key = [before[j + k] for k in range(n)]
                        value = [after[j + k] for k in range(n)]

                        '''   
                        if tuple(key) not in d:
                            d[tuple(key)] = value 
                        '''

                        if tuple(key) in d:
                            if tuple(value) in d[tuple(key)]:
                                d[tuple(key)][tuple(value)] += 1
                            else:
                                d[tuple(key)][tuple(value)] = 1
                        else:          
                            d[tuple(key)] = {tuple(value):1}   

    
                    # Clear arrays　　
                    before = []   
                    after = []

            train_ext.close()    
    
    #### Modify dict ####             
    # Find the max cnt in dictionary and modify
    for key, v_dict in d.items():
        max_key = max(v_dict, key=lambda k: v_dict[k])
        d[key] = list(max_key)
    

    return d


    
def ngram_dict_train(train, n, DATA_INPUT_PATH):

    d = {}

    #  Save n-gram dict from training set
    print("   Training set")
    grouped = train.groupby('sentence_id')

    # For each sentence
    for name, group in grouped:
        before = group['before'].values.tolist()
        after  = group['after'].values.tolist() 

        # For tokens
        for j in range(len(before) - n + 1):
            key = [before[j + k] for k in range(n)]
            value = [after[j + k] for k in range(n)]
            
            if tuple(key) in d:
                if tuple(value) in d[tuple(key)]:
                    d[tuple(key)][tuple(value)] += 1
                else:
                    d[tuple(key)][tuple(value)] = 1
            else:          
                d[tuple(key)] = {tuple(value):1}              

                
    #### Modify dict ####             
    # Find the max cnt in dictionary and modify
    for key, v_dict in d.items():
        max_key = max(v_dict, key=lambda k: v_dict[k])    
        d[key] = list(max_key)

        
    #### Save dict to pickle ####
    # with open('d_ru_' + 'n' + 'gram.pickle', 'wb') as f:
    #    pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)        
        
    return d



def cover_output(d, n, test):
    # print("   Covering old output")
    before = test['before'].values.tolist()
    token_id = test['token_id'].values.tolist()
    after = test['after'].values.tolist()

    for j in range(0, len(test) - n + 1): 
        # n-grams to check
        key = [before[j + k] for k in range(n)]
        key = tuple(key)

        # Need to check token_id is 0 (tokens should be in on sentence)
        if key in d: #  and (not (0 in token_id[j+1: j+n])):  
            for k in range(n):
                after[j + k] = d[key][k]
                
    test['after'] = after    
        
    return test



if __name__ == '__main__':
# if True:    
    # Load inputs
    test, train, files = input_files(lan, file_to_cover, DATA_INPUT_PATH)

    # Loop through n-gram
    for i in range(2, ngram + 1):
        print('******************************')
        print('Working on ' + str(i) + '-gram')
        ### Generate dict
        
        # Ext data
        for f in files: 
            d = ngram_dict_ext(f, i, DATA_INPUT_PATH)

            # Cover original output
            test = cover_output(d, i, test)

        # Training set
        d = ngram_dict_train(train, i, DATA_INPUT_PATH)
        test = cover_output(d, i, test)

        # Save to output
        test[['id','after']].to_csv(file_output + str(i) + '.csv', index=False)
        print("   File saved") 
  
    print('done')



