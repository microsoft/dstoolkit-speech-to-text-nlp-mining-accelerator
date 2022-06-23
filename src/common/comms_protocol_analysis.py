import os, sys, time

# setup the current paths
sys.path.append('../')
sys.path.append('./')

from common.constants import *

# system related
import configparser
import logging
import pandas as pd
import numpy as np
import json
import glob
import shutil
#import datetime
from datetime import datetime #<-- Solves the Error in module 'datetime' has no attribute 'strptime' issue
import ast
from dotenv import load_dotenv, find_dotenv
import http.client, urllib.request, urllib.parse, urllib.error, base64
from collections import OrderedDict
import pickle
#import matplotlib.pyplot as plt
import argparse
#from argparse import ArgumentParser
import shutil

# 5. Communication Protocol Analysis
# Class to perform the communication protocol analaysis of the transcribed text to extract key fields
# including callsigns, radio-check, CASEVAC analysis

class CommsProtocolAnalysis():
    """
    Class to perform various functions for the communications protocol to
    extract key insights of interest includiong communication classifications 
    such as radio-check, CASEVAC, etc, callsigns, sucessfult messaging, etc
    """

    def __init__(self):
        super().__init__() # inherit if applicable
       
        self.comms_development_dict_all = dict({'documents':list()})
        self.comms_analysis_dict_all = dict({'documents':list()})
        
        # create the audio file grouping dataframe
        self.audio_group_df = pd.DataFrame(columns=['id', 'group_id', 'group_uniqueness'])
        self.audio_group_df['id'] = self.audio_group_df['id'].astype(str)
        self.audio_group_df['group_id'] = self.audio_group_df['group_id'].astype(int)
        self.audio_group_df['group_uniqueness'] = self.audio_group_df['group_uniqueness'].astype(int)

        self.comms_analysis_dataframe_all = pd.DataFrame()
        
    def radioCheckAudioGrouping(self, transcript_dict):
        """
        Function to group the sequential set of radio check audio files
        """
        try:
            # initialise these local variables
            end_time_prev = None
            prev_index = 0  
            temp_dict = dict()
            group_id = 1

            # get the transcripted dictionary
            transcript_list = transcript_dict.get('documents')

            # loop through each ausio file and attemp tp group then into groups
            # based on the time difference of capture
            for index, transcript in enumerate(transcript_list):
                start_time = datetime.strptime(transcript.get('start_time'), '%H:%M:%S')
                end_time = datetime.strptime(transcript.get('end_time'), '%H:%M:%S')

                # get id
                temp_dict['id'] = transcript.get('id')

                # initialise for fisrt one in this loop
                # note: avoiding a group '0'
                if (index == 0): 
                    temp_dict['group_id'] = group_id

                else:
                    if end_time_prev is not None:
                        # get time difference from previous audio file
                        diff = (start_time - end_time_prev).total_seconds()
                        
                        # check if difference is less thana value
                        if( diff < 20):
                            temp_dict['group_id'] = int(prev_index)        
                        else:
                            group_id +=1
                            temp_dict['group_id'] = int(group_id)
                            prev_index = group_id
                
                # append    
                self.audio_group_df = self.audio_group_df.append(temp_dict, ignore_index=True)

                # also add the uniqueness of the grouping to aid in the comms analysis
                self.audio_group_df['group_uniqueness'] = self.audio_group_df['group_id'].map(self.audio_group_df['group_id'].value_counts())
                
                # store for next loop
                end_time_prev = end_time

        except Exception as error:
            print(f'Error in {error}')


    def radioCheckCommsDevelopment(self, definition_dict, transcript_dict):
        """
        Function to perform the radio check communication development
        and update the comms analysis dictionary
        """
        try:
            
            # temp dictionary to store intermediate values of the analysis
            comms_development_dict = dict()

            # obtain the transcripted dictionary
            transcripted_list = transcript_dict.get('documents')
            
            # obtain the individual keys from the radio-check corpus
            call_signs = definition_dict.get('call-signs')
            signal_readability = definition_dict.get('signal-readability')
            radio_check = definition_dict.get('initiation')
            message_end = definition_dict.get('message-end')
            comms_end = definition_dict.get('comms-end')

            # loop through the transcripted list and search for keys
            for transcript in transcripted_list:
                transcribed_text = transcript.get('text')
                
                # add the audio and group ids & add to this dictionary
                audio_id = transcript.get('id')
                group_id = int(self.audio_group_df[self.audio_group_df['id'] == audio_id]['group_id'])
                group_uniqueness = int(self.audio_group_df[self.audio_group_df['id'] == audio_id]['group_uniqueness'])

                # update some additional fields from the original dictionary
                comms_development_dict['id'] = audio_id
                comms_development_dict['group_id'] = group_id
                comms_development_dict['group_uniqueness'] = group_uniqueness
                
                comms_development_dict['start_date'] = transcript.get('start_date')
                comms_development_dict['start_time'] = transcript.get('start_time')
                comms_development_dict['end_date'] = transcript.get('end_date')
                comms_development_dict['end_time'] = transcript.get('end_time')
                comms_development_dict['text'] = transcript.get('text')
                comms_development_dict['duration'] = transcript.get('duration')
                comms_development_dict['offset'] = transcript.get('offset')
                comms_development_dict['confidence'] = transcript.get('confidence')
                
                # also add class truth (if exists)
                if transcript.get('ground_truth_class') is not None:
                    comms_development_dict['truth_classification'] = transcript.get('ground_truth_class') 
                else:
                    print('ground_truth_class not valid')

                # initlaise temp variables
                call_sign_list = list(['none']) 
                first_instance=False

                # search for call signs
                for call_sign in call_signs:
                    if transcribed_text.find(call_sign, 0, len(transcribed_text)) >= 0:
                        if not first_instance:
                            call_sign_list.remove('none')
                            call_sign_list.insert(0, call_sign)
                            first_instance=True
                        else:
                            call_sign_list.append(call_sign)
        
                # update the temp disctionary
                comms_development_dict['call-signs'] = call_sign_list

                # signal readability and its variantes
                for signal in signal_readability:
                    
                    # get also the varinat
                    signal_variant = signal.replace('and ', '')
                    
                    if (transcribed_text.find(signal, 0, len(transcribed_text)) >= 0 or 
                        transcribed_text.find(signal_variant, 0, len(transcribed_text)) >= 0):
                        comms_development_dict['signal-readability'] = signal
                        break
                    else:
                        comms_development_dict['signal-readability'] = 'none'   

                # message end flag
                if transcribed_text.find(message_end, 0, len(transcribed_text)) >= 0:
                    comms_development_dict['message-end'] = True

                else:
                    comms_development_dict['message-end'] = False 

                # comms end flag
                if transcribed_text.find(comms_end, 0, len(transcribed_text)) >= 0:
                    comms_development_dict['comms-end'] = True

                else:
                    comms_development_dict['comms-end'] = False

                # message classification - in this case radio check
                if transcribed_text.find(radio_check, 0, len(transcribed_text)) >= 0:
                    comms_development_dict['ML_classification'] = MESSAGE_CLASSIFICATION_GROUP[0]
                else:
                    comms_development_dict['ML_classification'] = 'none'     

                # update the main dictionary
                self.comms_development_dict_all['documents'].append(comms_development_dict)   
                comms_development_dict = dict()
            
        except Exception as error:
            print(f'Error in {error}')  

    def radioCheckCommsAnalysis(self):
        """
        Function to analyse the radio cehck communications and extract the
        key phrases and insights
        """
        try:
            # temp dictionary to store intermediate values of the analysis
            comms_analysis_dict = dict()

            # get the list of comms development dictionaries
            comms_dict_list = self.comms_development_dict_all.get('documents')

            # loop through the comms dictionary and assign the values
            for comms_dict in comms_dict_list:
                  
                # 1. analyse callsigns
                #---------------------
                # get call sign list
                call_sign_list = comms_dict.get('call-signs')
                call_sign_len = len(call_sign_list)

                if call_sign_len >= 1:

                    # 'probable' that first element is receiver
                    call_sign_receiver = call_sign_list[0]

                    # observe the remaining list - if only singular
                    # then sender didnt get trabscibed
                    if call_sign_len == 1:
                        call_sign_sender = 'none'

                    # 'probable' that second element is sender   
                    else:
                        call_sign_sender = call_sign_list[1]

                # could not transcribe ither receiver or sender
                else:
                    call_sign_sender = 'none'
                    call_sign_receiver = 'none'

                # append to the dictionary
                comms_analysis_dict.update(comms_dict)
                comms_analysis_dict['call-sign-receiver'] = call_sign_receiver
                comms_analysis_dict['call-sign-sender'] = call_sign_sender

                # append to main
                self.comms_analysis_dict_all.get('documents').append(comms_analysis_dict)
                comms_analysis_dict = dict()

        except Exception as error:
            print(f'Error in {error}') 
    

    def commsAnalysisDataframe(self):
        """
        Function to convert the comms analysis dictionary to dataframe
        for visulaisation ans serving purposes
        """
        try:
            # initliase the dataframe
            self.comms_analysis_dataframe_all = pd.DataFrame()

            # loop through the comms analysis dictionary and append to dataframe
            for comms_analysis_dict_list in self.comms_analysis_dict_all.get('documents'):
                temp_df = pd.DataFrame.from_dict(comms_analysis_dict_list, orient='index')
                temp_df = temp_df.transpose()
                self.comms_analysis_dataframe_all = self.comms_analysis_dataframe_all.append(temp_df)

        except Exception as error:
            print(f'Error in {error}') 
    
    def groupMatching(self, class_message_item, class_item):
        """
        Function (private) to return the classification based on the group item
        """
        group_class_item = class_message_item['ML_classification']
        call_signs_list = class_message_item['call-signs']
        call_sign_complete = False

        # also obtain the call-signs where a sender and reciver is acknoledged
        call_sign_len = max(len(call_signs) for call_signs in call_signs_list)
        
        # length of 2 indicate both sender and receiver
        if call_sign_len > 1:
            call_sign_complete = True
        else:
            call_sign_complete = False

        # check for radio check in the group and also its sub_class
        if any(group_class_item == class_item):
            classification = class_item
            
            # check for call-sign completeness
            if call_sign_complete:
                sub_class = f'{class_item}-PASS'
            else:
                sub_class = f'{class_item}-FAIL'
            
        else:
            classification = 'none'
            sub_class = 'none'
        
        #print(success)
        return  [classification, sub_class]


    def commsClassification(self, message_class):
        """
        Function to classify the communication messaging
        """
        try:
            
            # initliase a temporary dataframe
            grouping_df = pd.DataFrame()

            # apply classification by the group_id
            grouping_df = pd.DataFrame(self.comms_analysis_dataframe_all.groupby('group_id')[['ML_classification', 
                                        'message-end', 
                                        'call-signs']].apply(lambda class_message_item: self.groupMatching(class_message_item, message_class)))
            
            # prepare the grouping dataframe
            grouping_df.reset_index(inplace=True)
            grouping_df.columns = ['group_id', 'classification_subclass']

            # plit the return value from lamda function
            grouping_split_df = pd.DataFrame(grouping_df['classification_subclass'].tolist(), columns=['ML_classification', 'sub_class'])

            # rejoin the split dataframe
            grouping_df = pd.concat([grouping_df['group_id'], grouping_split_df], axis=1)

            # merge with main dataframe
            self.comms_analysis_dataframe_all = self.comms_analysis_dataframe_all.merge(grouping_df, left_on='group_id', right_on='group_id', 
                                                                                        suffixes=('_x', '')).filter(regex='^(?!.*_x)')
            # now analyse the truth against the ml classified messages
            if 'truth_classification' in self.comms_analysis_dataframe_all.columns:
                self.comms_analysis_dataframe_all['identified'] = np.where(self.comms_analysis_dataframe_all['truth_classification'] 
                                                            == self.comms_analysis_dataframe_all['ML_classification'], True, False)
            
        except Exception as error:
            print(f'Error in {error}') 
    

    def saveCommsAnalysisResults(self, datastore, file_path, target_path):
        """
        Function to save the results of the comms analysis which includes
        the visualisation dataframe
        """
        try:
            
            if self.comms_analysis_dict_all is not None:
                # Save the comms analysis as Serialize data into file:
                json.dump(self.comms_analysis_dict_all, open(f'{file_path}all_comms_analysis_dict.json', 'w' ) )

                # also save the dataframe
                if self.comms_analysis_dataframe_all is not None:
                    self.comms_analysis_dataframe_all.to_csv(f'{file_path}{COMMS_ANALYSIS_FILENAME}', sep =',', header=True, index=False)

                # also save to datastore (*Optional to allow use in pipeline)
                if(datastore is not None):
                    # also save in the registred datastore (reflected in the Azure Blob)
                    results_list = []
                    for path, subdirs, files in os.walk(file_path):
                        for name in files:
                            results_list.append(os.path.join(path, name))

                    #upload into datastore
                    datastore.upload_files(files = results_list, relative_root=file_path, 
                                            target_path= target_path, overwrite=True)   
                    print(f'Upload NLP results to wrting results for {datastore.name}')
                else:
                    print('No datastore provided to upload results.')

            else:
                raise Exception('Error. Comms_analysis_dict_all empty or not provided')
                    
        except Exception as error:
            print(f'Error in {error}') 
