#The Azure Cognitive services - Speech will be used in conjuncton with custom ML models to extract the positive interactions

# setup the current paths
import os, sys
currentDir = os.path.dirname(os.getcwd())
sys.path.append(currentDir)

# system related
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import ast
from dotenv import load_dotenv, find_dotenv
import jiwer
from pathlib import Path

# Azure Cognitive Services
import azure.cognitiveservices.speech as speechsdk

# import from common setups & environments
from common.constants import *

print('Loading environmental variables', load_dotenv(find_dotenv(ENVIORNMENT_FILE)))

class AzureCognitiveSpeechServices():
    """
    This class focuses on the Azure cognitive services for translating commands to lixical text
    and perform furthe NLP processing and ML analysis
    """
    def __init__(self, speech_key, location, validate, filtered):
        super().__init__() # inherit if applicable
        
        self.speech_key = speech_key
        self.location = location
        self.validate = validate
        self.filtered = filtered
        self.__file_flag = None
        self.speech_recognizer = None
        self.audio_file = None
        self.audio_file_path = None
        self.audio_file_id = None

        self.__prev_time = None

        # speech results
        self.text_results = []
        self.property_results = []
        self.done = False
        self.batch_result = None
        self.property_results_batch = []

        self.session_property_dict = dict({'Id':list(), 'RecognitionStatus':list(), 'Offset':list(), 
                                            'Duration':list(), 'DisplayText':list(), 'Lexical':list(), 
                                            'Confidence':list(), 'Latency': list()})

        # transcripted results - hypothesis
        self.transcripted_dict_all = dict({'documents':list()})
        self.transcript = None
        self.transcript_display_text = None

        # dataframe for visulaisation and serve layer
        self.transcripted_dataframe_all = pd.DataFrame()

        self.final_session_text = None
        self.char_exclude_list = [',', '"', '.', "'", ';', '?', ']', '[']

        # custom corpus dictionary
        self.custom_corpus = dict()

        # if validation is required then set them up
        if validate:
            # ground truth transcriptions
            self.transcripted_truth_dict_all = dict({'documents':list()})
            
            # Performance dataframe
            self.transcript_performance_df = pd.DataFrame(columns=['id', 'wer', 'mer', 'wil', 'wip', 'hits',
                                            'substitutions', 'deletions', 'insertions', 'wacc'])
        else:
            self.transcripted_truth_dict_all = None
            
            # Performance dataframe
            self.transcript_performance_df = None

        # set up for either filtered audio or not
        if self.filtered:
            self.__file_flag = '_filtered'
        else:
            self.__file_flag = ''


    def configSpeechtoText(self, audio_file, file_path, add_phrases, dictionary):
        """
        Function to configure the Azure cognitive services speech services
        """
        try:
            self.audio_file = audio_file
            self.audio_file_path = f'{file_path}{self.audio_file}'
            
            # also setup the id to save the results in JSON format
            self.audio_file_id = self.audio_file[:self.audio_file.find('.wav')]

            # Setting up the speech congiuration class. The keys are read form the .env file
            # This file will be hidden for the git push
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.location)

            # settings
            speech_config.speech_recognition_language="en-US"
            #speech_config.request_word_level_timestamps()
            speech_config.enable_dictation()
            speech_config.output_format = speechsdk.OutputFormat(1)

            # Setting up the audio configuration that points to an audio file.
            audio_input = speechsdk.audio.AudioConfig(filename=self.audio_file_path)

            # setting the speech recognizer
            self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
        
            # add key phrases - if required
            phrase_list_grammar = speechsdk.PhraseListGrammar.from_recognizer(self.speech_recognizer)

            if add_phrases:
                self.addPhrases(phrase_grammer=phrase_list_grammar, dictionary=dictionary)
                 
            
        except Exception as error:
            print(f'Error in {error}')

    def configCorpus(self, dictionary, asset_file_path, key_phrases_file_path, other_file_path):
        """
        Function to configure the custom corpus for boosting ML 
        """
        try:
            # initialise
            asset_name_list = list()
            key_phrases_list = list()
            other_key_phrases_list = list()

            # get the asset name key phrases
            temp_df = pd.read_csv(asset_file_path, encoding = 'unicode_escape', engine ='python').reset_index(drop=True)
            asset_name_list = temp_df.iloc[:,0].to_list()  

            # get the key phrases
            temp_df = pd.read_csv(key_phrases_file_path, encoding = 'unicode_escape', engine ='python').reset_index(drop=True)
            key_phrases_list = temp_df.iloc[:,0].to_list() 

            # get the other key phrases
            temp_df = pd.read_csv(other_file_path, encoding = 'unicode_escape', engine ='python').reset_index(drop=True)
            other_key_phrases_list = temp_df.iloc[:,0].to_list()

            # add to the main dictionary
            self.custom_corpus['assets'] = asset_name_list
            self.custom_corpus['interactional'] = key_phrases_list
            self.custom_corpus['others'] = other_key_phrases_list
   
            # update all other keys in original dictionary
            # update other fileds
            self.custom_corpus.update(dictionary)

        except Exception as error:
            print(f'Error in {error}')
           

    def addPhrases(self, phrase_grammer, dictionary):
        """
        Function to add key phrases to the transcription to enhance speech to text 
        based on the main dictionary
        """
        try:
            # extract all key-list values from the dictionary
            key_values_list = dictionary.values()

            # loop through the list and extract each individual element and add as key phrase
            for items in key_values_list:
                for item in items:
                    phrase_grammer.addPhrase(item)       

        except Exception as error:
            print(f'Error in {error}')


    def transcribeAudioFile(self):
        """
        Function to transcribe the audio file and priovide the formatted results
        """
        try:
            global done
            self.text_results = []

            def handle_final_result(evt):
                self.text_results.append(evt.result.text) 
                self.property_results.append(evt.result.properties)
                
            done = False

            def stop_cb(evt):
                print('CLOSING on {}'.format(evt))
                #self.speech_recognizer.stop_continuous_recognition()
                self.speech_recognizer.stop_continuous_recognition_async()
                global done
                done= True

            #Appends the recognized text to the all_results variable. 
            self.speech_recognizer.recognized.connect(handle_final_result) 

            #Connect callbacks to the events fired by the speech recognizer & displays the info/status
            self.speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
            self.speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
            self.speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
            self.speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
            self.speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))

            # stop continuous recognition on either session stopped or canceled events
            self.speech_recognizer.session_stopped.connect(stop_cb)
            self.speech_recognizer.canceled.connect(stop_cb)

            #self.speech_recognizer.start_continuous_recognition()
            self.speech_recognizer.start_continuous_recognition_async()

            while not done:
                time.sleep(0.5)
                
            print('transcribing completed')
            print('Transcription is\n', self.text_results) 

        except Exception as error:
            print(f'Error in {error}')

    def transcribeAudioFile_OneShot(self):
        """performs one-shot speech recognition with input from an audio file"""
          
        self.batch_result = self.speech_recognizer.recognize_once()


        # Check the result
        if self.batch_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: {}".format(self.batch_result.text))
        
        elif self.batch_result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(self.batch_result.no_match_details))
        
        elif self.batch_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = self.batch_result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))


    def transcribeResults(self, homophone_list=None):
        """
        Function to format thye results of the trabscribed audio file
        The results of the transcript may have multipl sessions of voide results
        """
        try:
            
            transcripted_dict = dict()
            session_properties_dict = dict({'Id':list(), 'RecognitionStatus':list(), 'Offset':list(), 
                                            'Duration':list(), 'DisplayText':list(), 'Lexical':list(), 
                                            'Confidence':list(), 'Latency': list()})
            best_lexical_list = []
            metadata = []
        
            # save the id irrespective of the transcription performance
            transcripted_dict['id'] = str(self.audio_file_id)
            
            # add further metadata - associated with the file name
            metadata = self.__decodeAudioFileName(self.audio_file_id)
            if len(metadata) > 4:
                transcripted_dict['start_date'] = metadata[0]
                transcripted_dict['start_time'] = metadata[1]
                transcripted_dict['end_date'] = metadata[2]
                transcripted_dict['end_time'] = metadata[3]
                transcripted_dict['message_duration'] = metadata[4]
            else:
                transcripted_dict['start_date'] = None
                transcripted_dict['start_time'] = None
                transcripted_dict['end_date'] = None
                transcripted_dict['end_time'] = None
                transcripted_dict['message_duration'] = None
            
            if len(self.property_results) > 0:
                for property_result in self.property_results:
                    property_values = property_result.values()
                    
                    #best text values - as a dict
                    best_text_values = list(property_values)[0]
                    best_text_dict = ast.literal_eval(best_text_values)
                                
                    # unpack the features of interest
                    session_properties_dict['Id'].append(best_text_dict.get('Id'))
                    session_properties_dict['RecognitionStatus'].append(best_text_dict.get('RecognitionStatus'))
                    session_properties_dict['Offset'].append(best_text_dict.get('Offset'))
                    session_properties_dict['Duration'].append(best_text_dict.get('Duration'))
                    session_properties_dict['DisplayText'].append(best_text_dict.get('DisplayText'))
                                
                    # also get the nested dictionary based on the best extracted text
                    if (best_text_dict.get('RecognitionStatus') == 'Success'):
                        best_lexical = best_text_dict.get('NBest')[0].get('Lexical')
                    
                        best_lexical_list.append(best_lexical)
                        session_properties_dict['Lexical'].append(best_lexical)
                
                        best_confidence = best_text_dict.get('NBest')[0].get('Confidence')
                        session_properties_dict['Confidence'].append(best_confidence)

                        # latency values - as a list
                        latency_values  = list(property_values)[1]
                        session_properties_dict['Latency'].append(latency_values)
            
                # keep session-level information
                self.session_property_dict = session_properties_dict

                # keep the final text results separatly
                self.transcript = str(best_lexical_list).translate({ord(x): '' for x in self.char_exclude_list})
                self.transcript_display_text = str(session_properties_dict['DisplayText']).translate({ord(x): '' for x in self.char_exclude_list})
                
                    # apply heuristics if required
                # note: word_replacement_list is dericed form the common file
                if homophone_list is not None:
                    for replace_text in homophone_list:
                        self.transcript = self.transcript.replace(replace_text[0], replace_text[1])  
                        self.transcript_display_text = self.transcript_display_text.replace(replace_text[0], replace_text[1])

                # add the parameters
                transcripted_dict['text'] = self.transcript
                transcripted_dict['display_text'] = self.transcript_display_text
                
                transcripted_dict['language'] = 'en'
                transcripted_dict['word_count'] = len(self.transcript.strip().split(' '))
                transcripted_dict['confidence'] = np.mean(self.session_property_dict.get('Confidence'))
                transcripted_dict['duration'] = (np.sum(self.session_property_dict.get('Duration'))) / 10000
                transcripted_dict['offset'] = self.session_property_dict.get('Offset')[0] / 10000
                
            else:
                transcripted_dict['text'] = None
            
            # also include the truth information if required
            if (self.transcripted_truth_dict_all is not None):
                # making sure it aligned wiht the truth
                transcripted_class_truth_list = list(self.transcripted_truth_dict_all.get('documents'))
            
                # extract the text - Note the dsp audio file will use a modified id
                if self.filtered:
                    truth_audio_id = str(self.audio_file_id[:self.audio_file_id.find(self.__file_flag)])
                else:
                    truth_audio_id = str(self.audio_file_id)

                class_list = [[d.get('ground_truth'), d.get('ground_truth_class')] for d in transcripted_class_truth_list if Path(d['id']).stem == truth_audio_id]
                
                if len(class_list) > 0:
                    transcripted_dict['ground_truth'] = class_list[0][0]  
                    transcripted_dict['ground_truth_class'] = class_list[0][1]  
                else: 
                    transcripted_dict['ground_truth'] = 'none'
                    transcripted_dict['ground_truth_class'] = 'none' 

            
            # append to a list of documents
            self.transcripted_dict_all['documents'].append(transcripted_dict)

            # need to apply this to ensure doesnt append to next session and cumulate
            self.property_results = []
            
        except Exception as error:
            print(f'Error in {error}')

    def transcribeResults_OneShot(self, apply_heuristics=False):
        """
        Function to format thye results of the trabscribed audio file
        The results of the transcript may have multipl sessions of voide results
        """
        try:
            
            transcripted_dict = dict()
            metadata = []
     
            # save the id irrespective of the transcription performance
            transcripted_dict['id'] = str(self.audio_file_id)
            
            # add further metadata - associated with the file name
            metadata = self.__decodeAudioFileName(self.audio_file_id)      
            transcripted_dict['date'] = metadata[0]
            transcripted_dict['time'] = metadata[1]
            transcripted_dict['time_diff'] = metadata[2]
            transcripted_dict['radio_id'] = metadata[3]


            # get the one shot results
            property_values = list(self.batch_result.properties.values())
            
            #best text values - as a dict
            best_text_values = property_values[0]
            best_text_dict = ast.literal_eval(best_text_values)

            # also get the nested dictionary based on the best extracted text
            if (best_text_dict.get('RecognitionStatus') == 'Success'):
                best_lexical = best_text_dict.get('NBest')[0].get('Lexical')
            
                # keep the final text results separatly
                self.transcript = str(best_lexical).translate({ord(x): '' for x in self.char_exclude_list})
                self.transcript_display_text = str(best_text_dict.get('DisplayText')).translate({ord(x): '' for x in self.char_exclude_list})

                # apply heuristics if required
                # note: word_replacement_list is dericed form the common file
                if apply_heuristics:
                    for replace_text in word_replacement_list:
                        self.transcript = self.transcript.replace(replace_text[0], replace_text[1])  
                        self.transcript_display_text = self.transcript_display_text.replace(replace_text[0], replace_text[1])


                # update the text components
                transcripted_dict['text'] = self.transcript
                transcripted_dict['display_text'] = self.transcript_display_text
        
                transcripted_dict['language'] = 'en'
                transcripted_dict['word_count'] = len(self.transcript.strip().split(' '))
                transcripted_dict['confidence'] = best_text_dict.get('NBest')[0].get('Confidence')
                transcripted_dict['duration'] = best_text_dict.get('Duration') / 10000
                transcripted_dict['offset'] = best_text_dict.get('Offset') / 10000

                if (self.transcripted_truth_dict_all is not None):
                    # making sure it aligned wiht the truth
                    transcripted_class_truth_list = list(self.transcripted_truth_dict_all.get('documents'))
                
                    # extract the text - Note the dsp audio file will use a modified id
                    if self.filtered:
                        truth_audio_id = str(self.audio_file_id[:self.audio_file_id.find(self.__file_flag)])
                    else:
                        truth_audio_id = str(self.audio_file_id)

                    class_truth = [d.get('interaction_truth_class') for d in transcripted_class_truth_list if d['id'] == truth_audio_id]
                    
                    if len(class_truth) > 0:
                        transcripted_dict['interaction_truth_class'] = class_truth[0]  
                    else: 
                        transcripted_dict['interaction_truth_class'] = 'none'
                
            else:
                transcripted_dict['text'] = None
            
            # append to a list of documents
            self.transcripted_dict_all['documents'].append(transcripted_dict)
            
        except Exception as error:
            print(f'Error in {error}')


    def __decodeAudioFileName(self, audio_file):
        """
        Function to devoce the audio file name to extract the date and time components
        """
        try:
            file_name_decoded = list()

            # split string into sections of interest  
            tokenized_filename = audio_file.split('_')

            # extract the dates and time
            from_date_time = tokenized_filename[1].split('-')
            to_date_time = tokenized_filename[2].split('-')

            # extract from date and time
            from_date = str(datetime.strptime(from_date_time[0], '%Y%m%d').date())
            from_time = datetime.strptime(from_date_time[1], '%H%M%S')

            # extract to date and time
            to_date = str(datetime.strptime(to_date_time[0], '%Y%m%d').date())
            to_time = datetime.strptime(to_date_time[1], '%H%M%S')
            
            # time difference in the message time stamp
            time_diff = (to_time - from_time).seconds

            # pack into list and return
            file_name_decoded.extend([from_date, str(from_time.time()), to_date, str(to_time.time()), time_diff])
        
        except Exception as error:
            print(f'Error in {error}') 
        
        finally:
            return file_name_decoded    

    def processDataframe(self):
        """
        Function to convert the transcripted dictionary to dataframe
        for visulaisation ans serving purposes
        """
        try:
            # loop through the comms analysis dictionary and append to dataframe
            for transcripted_dict_list in self.transcripted_dict_all.get('documents'):
                temp_df = pd.DataFrame.from_dict(transcripted_dict_list, orient='index')
                temp_df = temp_df.transpose()
                self.transcripted_dataframe_all = self.transcripted_dataframe_all.append(temp_df)
            
            # reset the index
            self.transcripted_dataframe_all.reset_index(drop=True, inplace=True)
        
        except Exception as error:
            print(f'Error in {error}')


    def saveTranscripts(self, datastore, file_path, target_path):
        """
        Function to save the transcripted results from the audio file
        The results are also stored in the Azure blob store
        """
        if(self.transcripted_dict_all is not None):
            try:                              
                # Save the transcripted dictionary (for NLP procesing) as Serialize data into file:
                json.dump(self.transcripted_dict_all, open(f'{file_path}{TRANSCRIBED_JSON_FILENAME}', 'w' ) )

                # save the disctionary also as dataframe for visualisation & serving purposes
                if self.transcripted_dataframe_all is not None:
                    self.transcripted_dataframe_all.to_csv(f'{file_path}{TRANSCRIBED_DATAFRAME_FILENAME}', sep =',', header=True, index=False)
                
                # the performance dataframe
                if(self.transcript_performance_df is not None):
                    self.transcript_performance_df.to_csv(f'{file_path}{TRANSCRIBED_PERFORMANCE_FILENAME}', index=False) 

                # also save in the registred datastore (reflected in the Azure Blob)
                if(datastore is not None):
                    results_list = []
                    for path, subdirs, files in os.walk(file_path):
                        for name in files:
                            results_list.append(os.path.join(path, name))

                    #upload into datastore
                    datastore.upload_files(files = results_list, relative_root=file_path, 
                                            target_path= target_path, overwrite=True)   
                    print(f'Upload results to wrting results for {datastore.name}')
                
                else:
                    raise Exception('Error in uploading results to datastore')
                    
                
            except Exception as error:
                print(f'Error in {error}')

        else:
            raise Exception('Error in mounted files to saving transcipted error')
    
    def processTranscriptsTruth(self, file_path):
        """
        Function to read the 'ground truth' transcriptions and prepare for
        perfromanbce analysis for the speech to text processing
        """
        ## transcribed file exist?
        if os.path.isfile(file_path):
            transcript_truth_df = pd.read_csv(file_path, sep=',', header=None, names=['audio_file', 'transcript_truth', 'transcript_class']).reset_index(drop=True)
        
            try: 
                transcripted_truth_dict = dict()
                # read the mounted file for processing

                # loop through each row and process it as a dictionary
                for audio_file in transcript_truth_df['audio_file']:
                    
                    # depending on the truth table format choose
                    transcripted_truth_dict['id'] = str(audio_file)

                    # perform some further processing wit the text
                    transcripted_truth_text = str(list(transcript_truth_df['transcript_truth'].loc[transcript_truth_df['audio_file'] == audio_file])[0])
                    
                    if len(transcripted_truth_text) > 0:
                        transcripted_truth_text = transcripted_truth_text.translate({ord(x): '' for x in self.char_exclude_list})
                        transcripted_truth_dict['ground_truth'] = transcripted_truth_text.lower()

                        # get also the transcript class
                        transcripted_truth_class = str(list(transcript_truth_df['transcript_class'].loc[transcript_truth_df['audio_file'] == audio_file])[0])
                        transcripted_truth_dict['ground_truth_class'] = transcripted_truth_class

                        transcripted_truth_dict['language'] = 'en'
                        transcripted_truth_dict['word_count'] = len(transcripted_truth_dict['ground_truth'].strip().split(' '))

                    else:
                        transcripted_truth_dict['ground_truth'] = 'nan'
                        transcripted_truth_dict['ground_truth_class'] = 'none'

                    self.transcripted_truth_dict_all['documents'].append(transcripted_truth_dict)
                    transcripted_truth_dict = dict()

            except Exception as error:
                print(f'Error in {error}')

        else:
            print(f'FILE {file_path} does not exist, and please prepare it.')


    def werCalculations(self, truth, hypothesis):
        """
        Function to calculate the 'Word Error Rate' (WER) based on the insertion, 
        deletions and substitutions
        """
        try:
            wer_result = None
            # setup the array
            d = np.zeros((len(truth) + 1) * (len(hypothesis) + 1), dtype=np.uint16)
            d = d.reshape((len(truth) + 1, len(hypothesis) + 1))

            # calculate levenstein distance
            for i in range(len(truth) + 1):
                for j in range(len(hypothesis) + 1):
                    if i == 0:
                        d[0][j] = j
                    elif j == 0:
                        d[i][0] = i

            for i in range(1, len(truth) + 1):
                for j in range(1, len(hypothesis) + 1):
                    if truth[i - 1] == hypothesis[j - 1]:
                        d[i][j] = d[i - 1][j - 1]
                    else:
                        substitution = d[i - 1][j - 1] + 1
                        insertion = d[i][j - 1] + 1
                        deletion = d[i - 1][j] + 1
                        d[i][j] = min(substitution, insertion, deletion)
           
            # return results
            wer_result = float(d[len(truth)][len(hypothesis)]) / len(truth)

        except Exception as error:
            print(f'Error in {error}')
        
        finally:
            return wer_result

    def werAnalysis(self):
        """
        Function to perform the 'Word Error Rate' (WER) of the trabscripted results
        using ground truth and the result fro the speech-to-text analysis (hypothesis)
        """
        try:
            hypothesis = None
            # search the text components of the dictioonaries for both truth and the hypothesis
            transcripted_truth_list = list(self.transcripted_truth_dict_all.get('documents'))
            
            # making sure it aligned wiht the truth
            transcripted_hypothesis_list = list(self.transcripted_dict_all.get('documents'))
            
            for transcripted_hypothesis in transcripted_hypothesis_list:

                # get the audio id
                audio_file_id = transcripted_hypothesis.get('id')

                # extract the text - Note the dsp audio file will use a modified id
                if self.filtered:
                    truth_audio_id = str(audio_file_id[:audio_file_id.find(self.__file_flag)])
                else:
                    truth_audio_id = str(audio_file_id)

                truth = [d.get('ground_truth') for d in transcripted_truth_list if Path(d['id']).stem == truth_audio_id]

                # ensure the ruth is not null & then perofrm analysis
                if ((len(truth) > 0 ) and (len(truth[0]) > 0)):
                    if truth[0] != 'nan':
                        truth = truth[0]
                    
                        # ensure correct id is utilised
                        if not self.filtered:
                            truth_audio_id = f'{truth_audio_id}' 
                        else:
                            truth_audio_id = f'{truth_audio_id}_filtered'   

                        hypothesis = [d.get('text') for d in transcripted_hypothesis_list if Path(d['id']).stem == audio_file_id][0]

                        if hypothesis is not None:
                            hypothesis = hypothesis.lower()
                        
                            # perform the measures & the wer
                            transcripted_measure = jiwer.compute_measures(truth, hypothesis=hypothesis)
                            transcripted_measure['wer_type2'] = self.werCalculations(truth=truth, hypothesis=hypothesis)
                            
                            # also add word accuracy 
                            transcripted_measure['wacc'] = 1 - transcripted_measure.get('wer')
                            transcripted_measure['wacc_type2'] = 1 - transcripted_measure.get('wer_type2') 
                            print(transcripted_measure)
                            # convert to dataframe and append to the overall performance dataframe
                            perf_df = pd.DataFrame.from_dict(transcripted_measure, orient='index')
                            perf_df = perf_df.transpose()
                            perf_df['id'] = audio_file_id

                            # append to overall    
                            self.transcript_performance_df = self.transcript_performance_df.append(perf_df)
                            self.transcript_performance_df.reset_index(drop=True)

            self.transcript_performance_df.reset_index(drop=True, inplace=True)
            
            print(f'Transcription WER performance is:', self.transcript_performance_df)
            
        except Exception as error:
            print(f'Error in {error}')