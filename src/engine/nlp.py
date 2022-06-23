import os, sys
import pandas as pd
import numpy as np
import shutil
from dotenv import load_dotenv, find_dotenv
import http.client, urllib.request, urllib.parse, urllib.error, base64
import argparse
import shutil


from dotenv import load_dotenv, find_dotenv
import http.client, urllib.request, urllib.parse, urllib.error, base64
from collections import OrderedDict
import pickle
#import matplotlib.pyplot as plt
import argparse
#from argparse import ArgumentParser
import shutil
import re

# audio file processing
from scipy.io import wavfile
from scipy import signal
import noisereduce as nr
from pydub import AudioSegment

# NLP libraries & utilities
import jiwer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


# audio file processing
from scipy.io import wavfile
from scipy import signal
import noisereduce as nr
from pydub import AudioSegment

# append the correct paths
currentDir = os.path.dirname(os.getcwd())
print(f'Current working directory: {currentDir}')
sys.path.append(currentDir)
sys.path.append(os.path.dirname(__file__) + './../')
sys.path.append(os.path.dirname(__file__) + '././')

# import from common setups & environments
from common.constants import *
from common.signal_processing import *
from common.general_utilities import *
from common.nlp_modelling import *

# import from common setups & environments
print('Loading environmental variables', load_dotenv(find_dotenv('.env')))

# extract the argument list 
# Copy the filtered audio files to the datastore using the OutputFileDatasetConfig object
# The OutputFileDatasetConfig object will be available as an input for the next step in the pipeline
parser = argparse.ArgumentParser()
parser.add_argument('--transcribed_data', type=str, dest='transcribed_data_dir', help='transcribed data folder mounting point')
parser.add_argument('--nlp_data', type=str, dest='nlp_data_dir', help='nlp data folder mounting point')
args = parser.parse_args()

transcribe_path = f'{args.transcribed_data_dir}' 
print(f'Output (Transcribed) Dataset Path is: {transcribe_path}')
nlp_path = f'{args.nlp_data_dir}' 
print(f'Output (nlp) Dataset Path is: {nlp_path}')


'''
# 5. NLP Analysis
#---------------------
nlp_results_path = f'{INFERENCE_PATH}{RESULTS_ASSESSED_PATH}{RECORDINGS_FOLDER}'
transcribed_file_path = f'{INFERENCE_PATH}{RESULTS_RECORDINGS_TRANSCRIBE_PATH}{TRANSCRIBED_JSON_FILENAME}'
corpus_file_path = f'{INFERENCE_PATH}{RESULTS_RECORDINGS_TRANSCRIBE_PATH}{CUSTOM_CORPUS_JSON_FILENAME}'
transcribed_df_file_path = f'{INFERENCE_PATH}{RESULTS_RECORDINGS_TRANSCRIBE_PATH}{TRANSCRIBED_DATAFRAME_FILENAME}'

read_from_memory = False

print('Performing the NLP processing, please wait ...')

# initialise the NLP service class
nlp = NLPModelling(cogs_url=COGS_URL, nlp_key=SPEECH_KEY)

# read the transcripted distionary, either from disk or memeory
if read_from_memory:
    transcripted_dictionary = speech.transcripted_dict_all
    custom_corpus = speech.custom_corpus
    transcripted_dataframe = speech.transcripted_dataframe_all
    transcript_df = speech.transcripted_dataframe_all
else:
    with open(transcribed_file_path, 'r') as read_file:
        transcripted_dictionary = json.load(read_file)
    
    with open(corpus_file_path, 'r') as read_file:
        custom_corpus = json.load(read_file)

    transcript_df = pd.read_csv(transcribed_df_file_path, encoding = 'unicode_escape', engine ='python').reset_index(drop=True)

if (transcripted_dictionary is not None):

    # perform the tokenization
    nlp.tokenizeTranscript(transcripted_dictionary)

    # perform the stop word filtering - passed the calss field as argument
    nlp.removeStopWords(nlp.tokenized_dict_all) 

    # extract the nouns from the tokenised dictionary
    nlp.nounExtraction(nlp.filtered_tokenized_dict_all)

    # perform key phrase extraction - on original transcribes
    nlp.keyPhraseExtraction(nlp_url=NLP_KEY_PHRASE_URL, body=transcripted_dictionary) 

    # perform non-domain specific NER extraction - on original transcribes
    nlp.nerExtraction(nlp_url=NLP_NER_URL, body=transcripted_dictionary) 

    # extract custom key interests - on original transcribes
    nlp.customKeyPhraseExtraction(definition_dict=custom_corpus, text_dictionary=transcripted_dictionary)
   
    # save the MLP results to local and datastore 
    # also send the transcripted dataframe to merge with NLP
    nlp.saveNLPResults(datastore=assessed_datastore, 
                        file_path = nlp_results_path, target_path = RESULTS_ASSESSED_PATH, 
                        transcript_dataframe=transcript_df)
else:
    print(f'Transcripted dictionary {transcripted_dictionary} is empty')

print('NLP processing completed')
'''

