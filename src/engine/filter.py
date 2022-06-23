# system related
import os, sys
import pandas as pd
import numpy as np
import shutil
from dotenv import load_dotenv, find_dotenv
import http.client, urllib.request, urllib.parse, urllib.error, base64
import argparse
import shutil

# audio file processing
from scipy.io import wavfile
from scipy import signal
import noisereduce as nr
from pydub import AudioSegment


# append the correct paths
sys.path.append(os.path.dirname(__file__) + './../')
sys.path.append(os.path.dirname(__file__) + '././')


# import from common setups & environments
from common.constants import *
from common.signal_processing import *
from common.general_utilities import *

# create temp directories
dsp_results_folders = f'{RESULTS_PATH}{RESULTS_DSP_PATH}'
transcripts_results_folder = f'{RESULTS_PATH}{RESULTS_TRANSCRIBE_PATH}'
utilConfig = GeneraltUtilities()
utilConfig.createTmpDir(dsp_results_folders)
utilConfig.createTmpDir(transcripts_results_folder)


# Setup input and output datasets
parser = argparse.ArgumentParser()
parser.add_argument('--raw_input_datapath', type=str, dest='raw_data_dir', help='Raw audio files data folder mounting point')
parser.add_argument('--processed_dir', type=str, dest='processed_data_dir', help='Processed audio files data folder mounting point')

args = parser.parse_args()

# Setup paths for use by script
raw_data_path = f'{args.raw_data_dir}'
processed_data_path = f'{args.processed_data_dir}'

# Create directories for paths
os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

print(f'Input raw audio file dataset path is: {raw_data_path}')
print(f'Output processed audio file dataset path is: {processed_data_path}')

# Extract all file names from the mounted input raw audio Dataset
all_files = os.listdir(raw_data_path)

# Now filter audio file list to only include .wav files
raw_audio_files = [x for x in all_files if x.endswith('.wav')][:5]
print(f'List of raw_audio_files to process: {raw_audio_files}')


# setup filtering
filterAudio = SignalProcessing()

# option to sapply butterworth filter
butterworth_filter = False

for raw_audio_file in raw_audio_files:
    print(f'Performing DSP (filtering) for audio file : {raw_audio_file}')
    
    # configure the filter
    filterAudio.configFilter(low_freq_cut=LOW_FREQ_CUTOFF, high_freq_cut=HIGH_FREQ_CUTOFF,
                            order=FILTER_ORDER)

    # read the audio file for the byte data and the sampling rate
    # through the class fields 
    filterAudio.readAudioFile(audio_file_name=raw_audio_file, mount_path=raw_data_path, stereo=True)  

    # apply the butterworth filter
    if butterworth_filter:
        filterAudio.butterworthFilter()

    # apply further noise reduction
    filterAudio.fftFilter(stationary=True)

    # save the audio filtered files locally
    filterAudio.saveAudioFiltered(filtered_audio_file_path=dsp_results_folders, volume=9)


for path, subdirs, files in os.walk(dsp_results_folders):
    for name in files:
        #if(name.endswith('_filtered.wav')): #<--- Removed to move the ground truth file as well.
        full_path = os.path.join(path,name)
        shutil.copy(full_path,processed_data_path)
