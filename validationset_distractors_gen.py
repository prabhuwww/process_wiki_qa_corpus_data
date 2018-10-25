# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:08:36 2018

Generate validation dataset with distractors using simple dev/validation 
dataset csv already generated using raw_data_processor.py

Note: Existing data from MICROSOFT RESEARCH WIKIQA CORPUS does not 
have distractors

Most of the preprocessing is based on links referenced in:
http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/

@author: Prabhu Selvaraj
"""

import pandas as pd
from raw_data_processor import get_final_data_file

def create_distractor_columns():
    '''Add 1 ground truth columns and 9 distractor columns to existing 
    validation set. 
    The 9 distractor columns are chosen at random.
    Note: Validation set will not have the rows with previous label 0
    '''
    # Get dev/validation csv file path name
    validation_file = get_final_data_file(dataset_type='dev')
    validation_distractor_file = validation_file.replace('.csv', '_dis.csv')
    
    # file is small (even small true labels) there should not be any memory issues
    validation_set_orig_df = pd.read_csv(validation_file, names=['Question', 'Answer', 'Label']) 
    validation_set_with_true_labels = validation_set_orig_df[(validation_set_orig_df['Label'] == 1)]
    
    validation_set_with_false_labels = validation_set_orig_df[(validation_set_orig_df['Label'] == 0)]
    validation_set_with_false_answers = validation_set_with_false_labels['Answer']

    
    # Add distractor columns
    distractor_cols = [ 'distractor_'+str(i) for i in range(1,9) ]

    data_length_true_label = len(validation_set_with_true_labels)
    validation_set_with_true_labels.index = range(1, data_length_true_label+1)
    print(len(validation_set_with_false_answers))
    print(len(validation_set_with_true_labels))

    distractor_df_input_dict={}

    for distractor_col in distractor_cols:
        print("Processing", distractor_col)
        pd_series_data = validation_set_with_false_answers.sample(n=data_length_true_label, axis=0)
        pd_series_data.index = range(1, data_length_true_label+1)
        print(pd_series_data)
        print(len(pd_series_data))
        print(type(pd_series_data))
        #return
        distractor_df_input_dict[distractor_col] = pd_series_data
    #print(distractor_df_input_dict)
    
    
    distractor_cols_df = pd.DataFrame(distractor_df_input_dict)
    print("distractor_cols_df", len(distractor_cols_df))
    print(distractor_cols_df.head())
    validation_set_with_true_labels = pd.concat([validation_set_with_true_labels, 
                                                 distractor_cols_df], sort=False, axis=1)
    
    print('Saving to ', validation_distractor_file)
    validation_set_with_true_labels.to_csv(validation_distractor_file, encoding='utf-8')

if __name__ == '__main__':
    create_distractor_columns()