# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:23:37 2018

Processes txt file data from MICROSOFT RESEARCH WIKIQA CORPUS (pre-downloaded)
and converts them to (train, test and dev) csv files 

Most of the preprocessing is based on links referenced in:
http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/

@author: Prabhu Selvaraj
"""

import os
#import unicodecsv
import csv
import logging
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import config_dev_env as config

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def get_raw_file(dataset_type='train'):
    ''' Get a raw source data file path & name based on the dataset type '''
    raw_file_name = config.SRC_FILE_NAME_MAP.get(dataset_type)
    if raw_file_name:
        raw_file_path = os.path.join(config.QA_CORPUS_DOWNLOAD_PATH, 
                                     raw_file_name)
    else:
        raise "%s not found" % raw_file_name 
    logging.info("Raw file path name:%s", raw_file_path)
    return raw_file_path

def get_final_data_file(dataset_type='train', custom_output_path=None):
    '''Get target csv file path + name based on source/raw file path+name'''
    raw_file_path = get_raw_file(dataset_type=dataset_type)
    if custom_output_path:
        # Just for safer reasons (considering input file is non csv) always update extension to csv
        output_file_path = custom_output_path[:-3] + 'csv'  
    else:
        output_file_path = raw_file_path[:-3] + 'csv'
    logging.info("Custom output Path:%s", output_file_path)
    return output_file_path

def process_raw_file(dataset_type='dev', stem=True, tokenize=True, 
                     lemmatize=True, custom_output_path=None):
    '''Process raw file datasource file, tokenize, stem, lemmatize and finally
    write the records to a csv file of the form: "Question, Answer, label"
    No shuffling is done here.
    (Ideas from https://github.com/rkadlec/ubuntu-ranking-dataset-creator/blob/master/src/create_ubuntu_dataset.py)
    
    Keyword Arguments:
        dataset_type -- Define train/dev/test (default dev)
        stem  -- nltk stem (defualt True)
        tokenize -- nltk word_tokenize  (default True)
        lemmatize -- nltk lemmatize  (default True)
        custom_output_path -- Custom output path including name like (/mydir/test.csv)
    Output:
        CSV file of the format in the same folder. Format "Question, Answer, label"
    '''
    logging.info('Processing raw file | Settingstokenize:%s, stem:%s, lemmatize:%s', 
                 tokenize, stem, lemmatize)
    # Get source/raw data file path & name
    raw_file_path = get_raw_file(dataset_type=dataset_type)
    
    # Get target file path
    output_file_path = get_final_data_file(dataset_type=dataset_type)  
    
    # Open source raw file for reading & processing and csv target file to write results
    with open(raw_file_path, 'r', encoding='utf8') as file_reader, \
            open(output_file_path, 'w', encoding='utf8', newline='') as csv_out:
        # For csv write object use newline='' to prevent new line "\n" at end
        csv_writer = csv.writer(csv_out)  # Create csv writer obj
        
        counter = 1
        # Read file line by line
        for line in file_reader:
            question, answer, label = line.split('\t')  # Split tab spaces

            if tokenize:
                # Tokenizing
                question = nltk.word_tokenize(question)
                answer = nltk.word_tokenize(answer)
                #print("question_tokenized", question)
                #print("answer_tokenized", answer)
            if stem:
                # Stemming
                question = [stemmer.stem(q) for q in question]
                answer = [stemmer.stem(a) for a in answer]
                #print("question_stemmed", question)
                #print("answer_stemmed", answer)
            if lemmatize:
                # Lemmatizing
                question = [lemmatizer.lemmatize(q) for q in question]
                answer = [lemmatizer.lemmatize(a) for a in answer]
                #print("question_lemmatized", question)
                #print("answer_tokenized", answer)
                
            # Make list of words to sentences again
            question = ' '.join(question)
            answer = ' '.join(answer)
            label = int(label)  # Errorout if not numeric
            # Write row using unicodecsv writer
            #logging.info('Writing:%s', [question, answer, label])
            csv_writer.writerow([question, answer, label])
            counter += 1
    logging.info('File processing for %s complete|Wrote %s lines to %s', 
                 raw_file_path, counter, output_file_path)
    

if __name__ == '__main__':
    #get_raw_file(dataset_type='dev')
    # Make sure you have the MICROSOFT RESEARCH WIKIQA CORPUS txt file downloaded
    # Update the config file with your own source dir path
    process_raw_file(dataset_type='dev')
    process_raw_file(dataset_type='train')
    process_raw_file(dataset_type='test')