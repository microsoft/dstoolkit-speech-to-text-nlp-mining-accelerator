import os, sys
from azureml.core import datastore
from azureml.core.dataset import Dataset
from azureml.core.runconfig import OutputData
from azureml.data import OutputFileDatasetConfig
from azureml.core.run import OutputDatasets, Run

from constants import *

# system related
import configparser
import logging
import pandas as pd
import numpy as np
import json
import glob
import shutil
import time
import datetime
import ast
from dotenv import load_dotenv, find_dotenv
import http.client, urllib.request, urllib.parse, urllib.error, base64
from collections import OrderedDict
import pickle
#import matplotlib.pyplot as plt
import argparse
#from argparse import ArgumentParser
import shutil

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
from common.general_utilities import *
from common.speech_services import *

# create a temp directory to store results with sub-foldrs
utilConfig = GeneraltUtilities()
transcripts_results_folder = f'{RESULTS_PATH}{RESULTS_TRANSCRIBE_PATH}'
utilConfig.createTmpDir(transcripts_results_folder)

# import from common setups & environments
print('Loading environmental variables', load_dotenv(find_dotenv('.env')))

SPEECH_KEY = os.environ.get('SPEECH_KEY')


# Copy the filtered audio files to the datastore using the OutputFileDatasetConfig object
# The OutputFileDatasetConfig object will be available as an input for the next step in the pipeline
parser = argparse.ArgumentParser()
parser.add_argument('--validate', type=str, dest='val', help='If Truth validation file exists, then set to true. Will default to False by default.')
parser.add_argument('--language', type=str, dest='lang', help='Language to be used when translating speech to text.')
parser.add_argument('--processed_data', type=str, dest='processed_data_dir', help='processed data folder mounting point')
parser.add_argument('--key_phrases_path', type=str, dest='key_phrases_dir', help='key phrases folder mounting point')
parser.add_argument('--transcribed_data', type=str, dest='transcribed_data_dir', help='transcribed data folder mounting point')
args = parser.parse_args()

validate_param = args.val
print(f'Validate parameter is: {validate_param}')
language_param = f'{args.lang}'
dataset_path = f'{args.processed_data_dir}' 
print(f'Input Dataset Path is: {dataset_path}')
key_phrases_path = f'{args.key_phrases_dir}' 
print(f'Input Dataset Path is: {key_phrases_path}')
transcribe_path = f'{args.transcribed_data_dir}' 
print(f'Output (Transcribed) Dataset Path is: {transcribe_path}')

# Make sure the dirs exists
os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
os.makedirs(os.path.dirname(key_phrases_path), exist_ok=True)
os.makedirs(os.path.dirname(transcribe_path), exist_ok=True)

# Check dataset_path to confirm it contains all files
dataset_path_files = os.listdir(dataset_path)
print(dataset_path_files)

# Script to translate filtered audio files to text

#setup
validate = False
filtered = False
syncronous = False
results_folder = MOUNT_PATH_INFERENCE_RECORDINGS
file_flag = '.wav'

print('Performing speech to text scripts. please wait ...')

# initilaise the speech to text
speech = AzureCognitiveSpeechServices(speech_key=SPEECH_KEY, location=LOCATION, 
                                        validate=validate, filtered=filtered)

'''
# prepare the ground truth transcripted text
if validate:
    truth_file_path = f'{MOUNT_PATH_INFERENCE_TRUTH}{RECORDINGS_TRUTH_INFERENCE_FILENAME}'
    speech.processTranscriptsTruth(truth_file_path)

asset_file_path = f'{MOUNT_PATH_INFERENCE_KEY_PHRASES}{ASSET_NAMES_FILENAME}'
key_phrases_file_path = f'{MOUNT_PATH_INFERENCE_KEY_PHRASES}{KEY_PHRASES_FILENAME}'
other_file_path = f'{MOUNT_PATH_INFERENCE_KEY_PHRASES}{OTHER_KEY_PHRASES_FILENAME}'
metadata_file_path = f'{MOUNT_PATH_INFERENCE_TRUTH}{METADATA_INFERENCE_FILENAME}'


# configure the custom ontology with corpus
speech.configCorpus(dictionary=positive_comms_dictionary, asset_file_path=asset_file_path, 
                                key_phrases_file_path = key_phrases_file_path, 
                                other_file_path = other_file_path)

# overwrite settings if require filtered audio files
if filtered:
    results_folder = f'{INFERENCE_PATH}{RESULTS_RECORDINGS_INFERENCE_DSP_PATH}'
    file_flag = f'_filtered{file_flag}'


if len(os.listdir(dataset_path)) is not 0:
    audio_files_filtered = os.listdir(dataset_path)

    audio_files_filtered = [x for x in audio_files_filtered if x.endswith(file_flag)]
    
    # loop through each mounted dataset and perform the speech-to-text process
    # the audio file to transcribe
    for audio_file_filtered in audio_files_filtered:
        print(f'Starting the filtered transcription for file: {audio_file_filtered}')
        
        speech.configSpeechtoText(audio_file_filtered, file_path=results_folder, add_phrases=True)

        if syncronous:
            #transcribe
            speech.transcribeAudioFile()
        
            # format results
            speech.transcribeResults(apply_heuristics=True)
        else:
            #transcribe
            speech.transcribeAudioFile_OneShot()
        
            # format results
            speech.transcribeResults_OneShot(apply_heuristics=True)
                
        # perform WER
        if validate:
            speech.werAnalysis(audio_file_filtered)
    
    # convert results t dataframe also
    speech.processDataframeWithMetadata(metadata_file_path)
    
    # save results to datastore 
    speech.saveTranscripts(datastore=None, file_path = transcribe_path, 
                            target_path= None)

else:
    print('could not transcibe the filtered audio files')

'''

