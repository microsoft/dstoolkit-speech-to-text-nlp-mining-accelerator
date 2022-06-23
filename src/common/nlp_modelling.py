# Script for NLP Modelloing

import os, sys
currentDir = os.path.dirname(os.getcwd())
sys.path.append(currentDir)

# system related
import pandas as pd

# system related
import json
from datetime import datetime 
import urllib.parse, urllib.error
from collections import OrderedDict
from itertools import groupby 
import pickle


# NLP libraries & utilities
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# sequential modelling
from difflib import SequenceMatcher

# require to download the nltk corpus
if not nltk.download('stopwords'):  
    nltk.download('stopwords')    

if not nltk.download('punkt'):
    nltk.download('punkt')  

if not nltk.download('averaged_perceptron_tagger'):
    nltk.download('averaged_perceptron_tagger')

if not nltk.download('wordnet'):
    nltk.download('wordnet')

if not nltk.download('omw-1.4'):
    nltk.download('omw-1.4')

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import mgrs

# Import custom packages
from common.constants import *
from common.general_utilities import *

# Retrieve Key
#TEXT_ANALYTICS_KEY = os.environ.get('TEXT_ANALYTICS_KEY')

class NLPModelling():
    """
    This class focuses on the Azure NLP cognitive services and advanced algorithms
    for extracting key entities from lexical text and perform 
    furthe NLP processing and ML analysis
    """
    def __init__(self, cogs_url, nlp_key):
        super().__init__() # inherit if applicable
        
        self.cogs_url = cogs_url
        self.nlp_key = nlp_key

        # store various NLP results
        self.key_phrases_dict_all = dict({'documents':list()})
        self.ner_dict_all = dict({'documents':list()})
        self.tokenized_dict_all = dict({'documents':list()})
        self.filtered_tokenized_dict_all = dict({'documents':list()})
        self.filtered_noun_tokenized_dict_all = dict({'documents':list()})
        self.free_form_key_interests_dict_all = dict({'documents':list()})
        self.custom_key_interests_dict_all = dict({'documents':list()})
        
        # global store - merge all results
        self.nlp_dict_all = dict({'documents':list()})
        self.nlp_dataframe_all = pd.DataFrame()

        # for topic modelling
        self.dictionary = None
        self.corpus = None
        self.num_topics = None
        self.num_topic_terms = None
        self.LDAmodel = None
        self.LDAdisplay = None
        self.lda_perplexity = None
        self.lda_coherence = None
        self.topics_dataframe_all = pd.DataFrame()
        self.tokenised_words = list()


    def tokenizeTranscript(self, text_dictionary):
        """
        Function to tokenize the lexical transcript
        This will be required to perform the NLP
        """
        transcripted_list = []
        tokenized_dict = dict()

        try:
            transcripted_list = text_dictionary.get('documents')
            
            for transcript_dict in transcripted_list:
                # get the text
                transcribed_text = str(transcript_dict.get('text')).lower()
                
                # obtain the tokeinzed words
                transcript_tokenized = []
                if transcribed_text is not None:
                    transcript_tokenized = word_tokenize(transcribed_text)  
               
                tokenized_dict['id'] = transcript_dict.get('id')
                tokenized_dict['tokenized_transcript'] = transcript_tokenized
                tokenized_dict['num_tokens'] = len(transcript_tokenized)

                # update the dictionary
                self.tokenized_dict_all['documents'].append(tokenized_dict)
                tokenized_dict = dict()
       
        except Exception as error:
            print(f'Error in {error}')


    def removeStopWords(self, text_dictionary):
        """
        Function to remove the stop words from the toknenized text
        """
        tokenized_list = []
        filtered_tokenized_dict = dict()
        try:
            # extract the stop words from the 'english' corpus
            stopWords_list = list(stopwords.words('english'))
            
            #get the tokenised list for all the audio transcriptions
            tokenized_list = text_dictionary.get('documents')

            # loop through the each list of dictionary and extract the tokens
            for tokenized_dict in tokenized_list:
                word_tokenize = tokenized_dict.get('tokenized_transcript')
                wordsFiltered = []
                
                for token in word_tokenize:
                    if token not in stopWords_list:
                        wordsFiltered.append(token)
                
                # append the temp dict
                filtered_tokenized_dict['id'] = tokenized_dict.get('id')
                filtered_tokenized_dict['filtered_tokenized_transcript'] = wordsFiltered
                filtered_tokenized_dict['num_filtered_tokens'] = len(wordsFiltered)
                
                #also sort the token frequency distribution
                token_fdist_dict = dict(nltk.FreqDist(wordsFiltered))
                sorted_fdist_dict = OrderedDict(sorted(token_fdist_dict.items(), reverse=True, key=lambda x: x[1]))
                filtered_tokenized_dict['token_fdisk'] = dict(sorted_fdist_dict)

                # update the main dictionary
                self.filtered_tokenized_dict_all['documents'].append(filtered_tokenized_dict)
                filtered_tokenized_dict = dict()
                        
        except Exception as error:
            print(f'Error in {error}') 

    
    def __getWordnetPos(tag):
        """
        Function (private) to get the tag
        """
        try:

            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN
        except Exception as error:
            print(f'Error in {error}')


    def lemmatizer(self, text):
        """
        Function to lemmatize the transcripted text in the dictionary
        """
        try:

            # Initialize the lemmatizer
            word_lemmatizer = WordNetLemmatizer()

            word_pos_tags = nltk.pos_tag(word_tokenize(text)) 
            a=[word_lemmatizer.lemmatize(tag[0], self.__getWordnetPos(tag[1])) for idx, tag in enumerate(word_pos_tags)] 
            return " ".join(a)

        except Exception as error:
            print(f'Error in {error}')
        finally:
            return " ".join(a)


    def nounExtraction(self, text_dictionary):
        """
        Function to extract the nouns form the tokenised dictionary
        """
        tokenized_list = []
        filtered_noun_tokenized_dict = dict()

        # setout the noun tags
        noun_tag_list = ['N', 'NNS', 'NNP', 'NNPS']

        try:
            #get the tokenised list for all the audio transcriptions
            tokenized_list = text_dictionary.get('documents')
            
            # loop through the each list of dictionary and extract the filtered tokens
            for tokenized_dict in tokenized_list:
                tokenised_words = tokenized_dict.get('filtered_tokenized_transcript')
            
                # prepare the noum list
                noun_list = list()
                non_noun_list = list()

                # tag the tokenised words (filtered) and tag (noun, verbs, etc)
                tagged_tokenised_words = nltk.pos_tag(tokenised_words)
                
                for (word, tag) in tagged_tokenised_words:
                    if tag in noun_tag_list: # If the word is a proper noun
                        noun_list.append(word)
                    else:
                        non_noun_list.append(word)    
                
                # append the temp dict
                filtered_noun_tokenized_dict['id'] = tokenized_dict.get('id')
                filtered_noun_tokenized_dict['filtered_noun_tokenized_transcript'] = noun_list
                filtered_noun_tokenized_dict['filtered_not_noun_tokenized_transcript'] = non_noun_list
                filtered_noun_tokenized_dict['num_filtered_noun_tokens'] = len(noun_list)
                filtered_noun_tokenized_dict['num_filtered_non_noun_tokens'] = len(non_noun_list)
                
                #also sort the token frequency distribution
                token_fdist_dict = dict(nltk.FreqDist(noun_list))
                sorted_fdist_dict = OrderedDict(sorted(token_fdist_dict.items(), reverse=True, key=lambda x: x[1]))
                filtered_noun_tokenized_dict['noun_token_fdisk'] = dict(sorted_fdist_dict)
                
                token_fdist_dict = dict(nltk.FreqDist(non_noun_list))
                sorted_fdist_dict = OrderedDict(sorted(token_fdist_dict.items(), reverse=True, key=lambda x: x[1]))
                filtered_noun_tokenized_dict['non_noun_token_fdisk'] = dict(sorted_fdist_dict)
                
                # update the main dictionary
                self.filtered_noun_tokenized_dict_all['documents'].append(filtered_noun_tokenized_dict)  
                filtered_noun_tokenized_dict = dict()
                        
        except Exception as error:
            print(f'Error in {error}')



    def keyPhraseExtraction(self, nlp_url, body):
        """
        Function to extract te key phrases from the text body
        This call is performed in batches
        """
        try:
            # call the nlp service via API request
            utilConfig = GeneraltUtilities()
            
            # send the request in batches
            # setup batch & temporary dictionaries
            batched_dictionary = dict({'documents':list()})
            temp_dictionary = dict({'documents':list()})

            # get the list of dictionaries form the body
            dictionary_list = body.get('documents')

            dictionary_size = len(dictionary_list)    
            # loop through the list and sent it in batches - eg: 5
            for i in range(0, dictionary_size, KEY_PHRASE_BATCH_SIZE):

                # prepare the sned dictionary
                batched_dictionary['documents'] = (dictionary_list[i: min(i+KEY_PHRASE_BATCH_SIZE, dictionary_size)])

                # send to the API call
                utilConfig.processRequest(url=self.cogs_url, cogs_service=nlp_url, key=self.nlp_key, body=batched_dictionary)

                # get the results as a class field - Note: converted to dictionary format from string
                temp_dictionary = json.loads(utilConfig.response_data)
                
                # append to the main dictionary
                self.key_phrases_dict_all['documents'].extend(temp_dictionary.get('documents'))
                batched_dictionary['documents']=[]
      
            # loop through the document 'ids' and add additional information
            keyPhrases_list = self.key_phrases_dict_all.get('documents')
            
            for keyPhrase_dict in keyPhrases_list:
                num_key_phrases = len(keyPhrase_dict.get('keyPhrases'))
                keyPhrase_dict['num_key_phrases'] = num_key_phrases
                self.key_phrases_dict_all.update(num_key_phrases = num_key_phrases)

        except Exception as error:
            print(f'Error in {error}')


    def nerExtraction(self, nlp_url, body):
        """
        Function to extract 'Named Entity Recognition' (NER) from text body
        This call is performed in batches
        """
        try:
            # call the nlp-NER service via API request
            utilConfig = GeneraltUtilities()

            # send the request in batches
            # setup batch & temporary dictionaries
            batched_dictionary = dict({'documents':list()})
            temp_dictionary = dict({'documents':list()})

            # get the list of dictionaries form the body
            dictionary_list = body.get('documents')

            dictionary_size = len(dictionary_list)    
            # loop through the list and sent it in batches - eg: 5
            for i in range(0, dictionary_size, NER_PHRASE_BATCH_SIZE):
                # prepare the sned dictionary
                batched_dictionary['documents'] = (dictionary_list[i: min(i+NER_PHRASE_BATCH_SIZE, dictionary_size)])

                # send to the API call
                utilConfig.processRequest(url=self.cogs_url, cogs_service=nlp_url, key=self.nlp_key, body=batched_dictionary)
                
                # get the results as a class field - Note: converted to dictionary format from string
                temp_dictionary = json.loads(utilConfig.response_data)

                # append to the main dictionary
                self.ner_dict_all['documents'].extend(temp_dictionary.get('documents'))
                batched_dictionary['documents']=[]

        except Exception as error:
            print(f'Error in {error}')    

    def freeFormExtraction(self, definition_dict, text_dictionary):
        """
        Function to perform the commas analysis for the free form text
        """
        try:
            # initialisation
            free_form_keyInterests_dict = dict()
            call_sign_list = list()

            # get the list of dictionaries form the body
            dictionary_list = text_dictionary.get('documents')
            
            
            # obtain the individual keys from the radio-check corpus
            call_signs = definition_dict.get('call-signs')
            radio_check = definition_dict.get('initiation')
            message_end = definition_dict.get('message-end')
            comms_end = definition_dict.get('comms-end')
            
            # loop throug the dictionary list extract the text component for further analysis for each audio
            for dictionary in dictionary_list:
                
                # extract the text component
                transcribed_text = dictionary.get('text')
                
                # setting the call sign find flag
                call_sign_list = list(['none', 'none']) 
                first_instance=True

                # search for call signs
                for call_sign in call_signs:
                    if transcribed_text.find(call_sign, 0, len(transcribed_text)) >= 0:
                        call_sign_list_all.append(call_sign)
                        if first_instance:
                            call_sign_list[0] = call_sign
                            first_instance=False
                            
                        else:
                            call_sign_list[1] = call_sign

                # append the temp dict
                free_form_keyInterests_dict['id'] = dictionary.get('id')
                free_form_keyInterests_dict['call_sign_1'] = call_sign_list[0]
                free_form_keyInterests_dict['call_sign_2'] = call_sign_list[1]
                free_form_keyInterests_dict['call_signs'] = call_sign_list_all

                # other key interests
                #if transcribed_text.find(message_end, 0, len(transcribed_text)) >= 0:
                #    free_form_keyInterests_dict['message-end'] = True
                # Defaults message-end to False, unless overwritten when found in loop
                free_form_keyInterests_dict['message-end'] = False
                free_form_keyInterests_dict['comms-end'] = False

                for word in transcribed_text.split():
                    if message_end == word:    
                        free_form_keyInterests_dict['message-end'] = True
                    if comms_end == word:
                        free_form_keyInterests_dict['comms-end'] = True
                
                # comms end flag
                #if transcribed_text.find(comms_end, 0, len(transcribed_text)) >= 0:
                #    free_form_keyInterests_dict['comms-end'] = True

                #else:
                #    free_form_keyInterests_dict['comms-end'] = False

            
                # message classification - in this case radio check
                if transcribed_text.find(radio_check, 0, len(transcribed_text)) >= 0:
                    free_form_keyInterests_dict['ML_classification'] = MESSAGE_CLASSIFICATION_GROUP[0]
                else:
                    free_form_keyInterests_dict['ML_classification'] = 'none'
            
                # update the main dictionary
                self.free_form_key_interests_dict_all['documents'].append(free_form_keyInterests_dict)   
                free_form_keyInterests_dict = dict()
                call_sign_list_all = list()
                
        except Exception as error:
            print(f'Error in {error}')

    def __mgrsToLatLong(self, mgrs_code):
        """
        Function to convert MGRS code to latitude and longitude
        """
        try:
            # initilaise an mgrs object
            mgrs_object = mgrs.MGRS()

            if mgrs_code is not None:
                latitude, longitude = mgrs_object.toLatLon(mgrs_code)
            
            else:
                raise('Valid MGRS code is not povided')
        except Exception as error:
            print(f'Error in {error}')

        finally:
                return latitude, longitude

    
    def customKeyPhraseExtraction(self, text_dictionary, main_dictionary, key_phrase_dictionary, word_to_num_dict):
        """
        Function to extract addional custom keyphrases of interest
        """
        try:
            # intialise main dictionary
            custom_key_phrase_dict = dict()
            
            # obtain the key phrase search categories of interest
            general_phrases = key_phrase_dictionary.get('general_phrases')

            # general phrases
            general_phrases_list = list()
            
            # get the list of dictionaries form the body
            dictionary_list = text_dictionary.get('documents')

            # loop through the dictionary list extract the text component for further analysis for each audio
            #-----------------------------------------------------------------------------------------------
            # also provide additional boolean found for each feature
            for dictionary in dictionary_list:
                
                # extract the text component
                transcribed_text = str(dictionary.get('text')).lower()
                
                # 1. general phrase extraction at complete paragraph level
                #---------------------------------------------------------
                # general phrases
                #----------------
                for general_phrase in general_phrases:
                    if transcribed_text.find(general_phrase, 0, len(transcribed_text)) >= 0:
                        general_phrases_list.append(general_phrase)

                # append the reaminder of phrases to the temp dict
                #-------------------------------------------------
                custom_key_phrase_dict['id'] = dictionary.get('id')

                # general phrases
                custom_key_phrase_dict['general_phrases'] = general_phrases_list
              
                # update the main dictionary & re-initialise for next time
                #---------------------------------------------------------
                self.custom_key_interests_dict_all['documents'].append(custom_key_phrase_dict)   
                
                custom_key_phrase_dict = dict()

        except Exception as error:
            print(f'Error in {error}')


    def createCorpus(self, text_dictionary, corpus_type):
        """
        Function to create the corpus & dictionary to perform topic modelling 
        of the transcribed text (tokenised)based on using LDA (Latent Dirichlet Allocation) 
        nearest-neighbor unsupervised algorithms
        """
        tokenized_list = []
        #tokenised_words = []   #<---- Nej's code doesnt do this. Need to check.
        try:
            # get the tokenized list 
            tokenized_list = text_dictionary.get('documents')

            # determine corpus type
            if corpus_type == 'noun':
                tokenized_type = 'filtered_noun_tokenized_transcript'
            else:
                tokenized_type = 'filtered_tokenized_transcript'
            print(f'tokenized_list: {tokenized_list}')
            # loop through each list of dictionary (for the audio id)
            for tokenized_dict in tokenized_list:
                word_tokenize = tokenized_dict.get(tokenized_type) 
                self.tokenised_words.append(word_tokenize)

            print(f'word_tokenize: {word_tokenize}')
            print(f'self.tokenised_words:{self.tokenised_words}')
            #print(f'self.tokenised_words:{tokenised_words}')
            # create the dictionary (mapping of words)
            self.dictionary = Dictionary(self.tokenised_words)
            print(f'Dictionary is: {self.dictionary}')
            # create corpus
            self.corpus = [self.dictionary.doc2bow(text) for text in self.tokenised_words]

        except Exception as error:
            print(f'Error in {error}') 


    def topicModelling(self, num_topics, num_topic_terms):
        """
        Function to perform topic modelling of the transcribed text (tokenised)
        based on using LDA (Latent Dirichlet Allocation) nearest-neighbor
        unsupervised algorithms
        """
        try:

            # initialise the number of topics to discover    
            self.num_topics = num_topics
            self.num_topic_terms = num_topic_terms
            tuple_list = list(tuple())
            temp_tuple = tuple()
            column_list = list()
            
            print(self.corpus)

            if (self.corpus is not None):
                self.LDAmodel = gensim.models.ldamodel.LdaModel(corpus=self.corpus, num_topics=self.num_topics, 
                                                                id2word=self.dictionary, 
                                                                passes=30, 
                                                                alpha='auto', 
                                                                per_word_topics=False,
                                                                update_every=0)

                # display the topics of interest
                topics = self.LDAmodel.print_topics(num_words=self.num_topic_terms)

                print('The topics of interest are:')    
                for count, topic in enumerate(topics):
                    print(f'Topic {count}: {topic}')
            
                # calculate the LDA model performance
                # Compute Perplexity - model performance
                self.lda_perplexity = self.LDAmodel.log_perplexity(self.corpus)
                
                # Coherence Score
                lda_coherence = gensim.models.ldamodel.CoherenceModel(model=self.LDAmodel, texts=self.tokenised_words, dictionary=self.dictionary, coherence='c_v')
                self.lda_coherence = lda_coherence.get_coherence()
               
                print('\nLDA Perplexity: ', self.lda_perplexity)  # a measure of how good the model is. lower the better.
                print('\nCoherence Score: ', self.lda_coherence)

                # also create the dataframe
                # get the topic lists
                topics_list = self.LDAmodel.get_topics()

                # loop through the topics and extract the tuples of terms and weights
                for topic_num, topic in enumerate(topics_list):
                    topic = sorted(topic, reverse=True)[:NUM_TOPIC_TERMS]
                    term_pairs_list = self.LDAmodel.get_topic_terms(topic_num)
                    
                    for term_pair in term_pairs_list:
                        temp_tuple = (self.dictionary[term_pair[0]], term_pair[1])
                        tuple_list.append(temp_tuple)

                    # append to dataframe and prepare the columns
                    self.topics_dataframe_all = pd.concat([self.topics_dataframe_all, pd.DataFrame(tuple_list)], ignore_index=False, axis=1)
                    column_list.extend((f'topic_{topic_num+1}_terms', f'topic_{topic_num+1}_weights'))
                    # init for next time
                    tuple_list = list(tuple())

                # add the colum names
                self.topics_dataframe_all.columns = column_list
                
            else:
                raise Exception('Error in corpus definition') 

        except Exception as error:
            print(f'Error in {error}')   


    def topicPlot(self,num_words, file_path):
        """
        Function to plot the results of the LDA topic modelling
        """
        try:
            if(self.LDAmodel is not None):
                topics = self.LDAmodel.print_topics(num_words=num_words)

                # displaying the topics of interest
                print('The topics of interest are:')    
                for count, topic in enumerate(topics):
                    print(f'Topic {count}: {topic}')

                # display the LDA topics
                self.LDAdisplay = pyLDAvis.gensim_models.prepare(self.LDAmodel, self.corpus, self.dictionary, sort_topics=True)
                #pyLDAvis.enable_notebook()
                #pyLDAvis.display(self.LDAdisplay)
                
                #also save the display as 'html'
                print('Saving LDAModel HTML file.')
                lda_display_file = f'{file_path}{MODEL_DISPLAY_FILENAME}'
                pyLDAvis.save_html(self.LDAdisplay, lda_display_file)

            else:
                raise Exception('Error lisplaying LDA model due to null model') 
              
        except Exception as error:
            print(f'Error in {error}')    

    def removeDictKeys(self, key_list, dictionary):
        """
        Function (private) to remove nultiple keys from dictionary
        """
        for key in key_list:
            if key in dictionary.keys():
                dictionary.pop(key) 
        return dictionary   

    def nerDataFrameConstruct(self):
        """
        Function (private) to construct the NER dataframe
        """
        try:
             # initialise some lists and dataframes - simpler than looping through
            remove_key_list = ['offset', 'length']
            key_columns = ['text','category','subcategory', 'confidenceScore']

            text_str = 'ner_text_'
            category_str = 'ner_category_'
            subcategory_str = 'ner_subcategory_'
            confidence_str = 'ner_confidence_'
            text_columns = [f'{text_str}1', f'{text_str}2', f'{text_str}3', f'{text_str}4', f'{text_str}5']
            category_columns = [f'{category_str}1', f'{category_str}2', f'{category_str}3', f'{category_str}4', f'{category_str}5']
            subcategory_columns = [f'{subcategory_str}1', f'{subcategory_str}2', f'{subcategory_str}3', f'{subcategory_str}4', f'{subcategory_str}5']
            confidenceScore_columns = [f'{confidence_str}1', f'{confidence_str}2', f'{confidence_str}3', f'{confidence_str}4', f'{confidence_str}5']

            main_ner_df = pd.DataFrame()
            ner_df = pd.DataFrame(columns=key_columns)

            # loop through each record and extract the NER from dictionary
            for row in range(len(self.nlp_dataframe_all)):

                # get the NER entities for the audio
                ner_dict_list = self.nlp_dataframe_all.loc[row,'entities']

                # check how many entitiy types have been recognised 
                # and get at most 5   
                num_ner_phrases = min(len(ner_dict_list), 5)

                # if exists then extract
                if num_ner_phrases > 0:
                    ner_dict_list = ner_dict_list[:num_ner_phrases]

                    # loop through all the entities within the audio extracted text
                    for ner_dict in ner_dict_list:
                        # remove some keys
                        ner_dict = self.removeDictKeys(remove_key_list,ner_dict)
                        # convert to dataframe
                        temp_df = pd.DataFrame.from_dict(ner_dict, orient='index').transpose()
                        ner_df = ner_df.append(temp_df)

                    # re-index column to aligh with merging
                    ner_df = ner_df.reindex(columns=key_columns)

                    ner_df = ner_df.melt().transpose().reset_index(drop=True)

                    ner_df.drop(index=ner_df.index[0], axis=0, inplace=True)
                    
                    # assign the correct columns
                    ner_df.columns = text_columns[:num_ner_phrases] + category_columns[:num_ner_phrases] + \
                    subcategory_columns[:num_ner_phrases] + confidenceScore_columns[:num_ner_phrases]

                    # append to the main ner dataframe
                    main_ner_df = main_ner_df.append(ner_df, sort=False).reset_index(drop=True)
                    main_ner_df.loc[row,'id'] = self.nlp_dataframe_all.loc[row,'id']
                    ner_df = pd.DataFrame()

                else:
                    main_ner_df.loc[row,'id'] = self.nlp_dataframe_all.loc[row,'id']
                    ner_df = pd.DataFrame()

        except Exception as error:
            print(f'Error in {error}')    
        
        finally:
            return main_ner_df


    def __nlpDataframe(self):
        """
        Function to convert the nlp dictionary to dataframe
        for visulaisation and serving purposes
        """

        try:
            # initialise
            string_null_list =  ['NaN' for item in range(4)]
            
            nlp_key_phrases = pd.DataFrame().astype(str)
            general_phrases_df = pd.DataFrame().astype(str)
          
            # column names list
            column_list = list()

            # first create a dataframe by looping through the dictionary and append to dataframe
            for nlp_dict_list in self.nlp_dict_all.get('documents'):
                temp_df = pd.DataFrame.from_dict(nlp_dict_list, orient='index').transpose()
                self.nlp_dataframe_all = self.nlp_dataframe_all.append(temp_df)

            # reset the index
            self.nlp_dataframe_all = self.nlp_dataframe_all.reset_index(drop=True)

            # loop through and get list of key phrases extracted for each audio 
            #------------------------------------------------------------------
            for row in range(len(self.nlp_dataframe_all)):

                # general key phrases
                #----------------
                # Generic Key phrase list (row based)
                key_phrase_list = self.nlp_dataframe_all.loc[row,'general_phrases'] 

                # check for length of list
                if type(key_phrase_list) is list and len(key_phrase_list) > 0:
                        num_key_phrases = min(len(key_phrase_list), 4)
                        key_phrase_list = key_phrase_list[:num_key_phrases]
                else:
                    key_phrase_list = string_null_list

                # update the dataframe
                general_phrases_df = general_phrases_df.append(pd.DataFrame(key_phrase_list).transpose().astype(str))
               
                # nlp key phrases
                #----------------
                # Generic Key phrase list (row based)
                key_phrase_list = self.nlp_dataframe_all.loc[row,'keyPhrases'] 

                # check for length of list
                if type(key_phrase_list) is list and len(key_phrase_list) > 0:
                        num_key_phrases = min(len(key_phrase_list), 4)
                        key_phrase_list = key_phrase_list[:num_key_phrases]
                else:
                    key_phrase_list = string_null_list

                # update the dataframe
                nlp_key_phrases = nlp_key_phrases.append(pd.DataFrame(key_phrase_list).transpose().astype(str))
            
            # general phrases
            column_list = list()
            for item in range(len(general_phrases_df.columns)):
                column_list.append(f'general_phrases_{item}')

            general_phrases_df.columns = column_list
            general_phrases_df = general_phrases_df.reset_index(drop=True)

            # nlp key phrases
            column_list = list()
            for item in range(len(nlp_key_phrases.columns)):
                column_list.append(f'nlp_keyphrase_{item}')

            # update the colum names
            nlp_key_phrases.columns = column_list
            nlp_key_phrases = nlp_key_phrases.reset_index(drop=True)


            # add to the main dataframe
            self.nlp_dataframe_all = pd.concat([self.nlp_dataframe_all,
                                            # scenario_df, 
                                            # scenario_pos_df, 
                                            # call_sign_df, 
                                            # casevac_stop_df,
                                            # serial_item_df, 
                                            # time_df, grid_df, 
                                            general_phrases_df, 
                                            # code_names_df,
                                            # vehicles_df, 
                                            # kinetics_df, 
                                            # offensive_df, 
                                            # direction_df,
                                            nlp_key_phrases], 
                                            ignore_index=False, axis=1)

        except Exception as error:
            print(f'Error in {error}')

    
    def __updateNLPDictionary(self, main_dict_list, dict_list):
        """
        Function (internal) to update the NLP dictionary
        and return the dictionary
        """
        temp_dict = dict({'documents':list( dict())})

        try:
            for index, main_dict in enumerate(main_dict_list): 
                for dictionary in dict_list:   
                    if main_dict.get('id') == dictionary.get('id'): 
                        main_dict.update(dictionary)
                            
                temp_dict['documents'].append(main_dict)
    
        except Exception as error:
            print(f'Error in {error}')
        
        finally:
            return temp_dict


    def processNLPResults(self, dictionary):
        """
        Function to process and combine all the previous dictionaries (in JSON)
        into the nlp dictinary
        """
        try:
            if len(dictionary) > 0:
               
                # make a copy of the completed tokenized dictionary
                self.nlp_dict_all = dictionary.copy()

                # get the list of dictionaries as needed
                keyPhrases_list = self.key_phrases_dict_all.get('documents')
                tokenized_list = self.filtered_tokenized_dict_all.get('documents')
                custom_key_phrase_list = self.custom_key_interests_dict_all.get('documents')
                ner_list = self.ner_dict_all.get('documents')

                # setup the dictionary list
                dictionary_list = [tokenized_list, keyPhrases_list, custom_key_phrase_list, ner_list]

                # loop through the list and update the main dictionary
                for dictionary in dictionary_list:
                    self.nlp_dict_all = self.__updateNLPDictionary(self.nlp_dict_all.get('documents'), dictionary)

                # Also process the dataframe
                # convert the dictionary to dataframe & save (internal function call)
                self.__nlpDataframe()
        
            else:
                raise Exception('Error in the dictionary provided')    

        except Exception as error:
            print(f'Error in {error}')   

    def sequenceMatching(self, sequence_dictionary):
        """
        Function to apply various comms analaysis, inclusing sequential analysis
        and apply as feature extraction 
        """
        try:
            # obtain thne various scenario protocol sequences
            if sequence_dictionary is not None and len(self.nlp_dict_all) > 0:
                radio_check_protocol = sequence_dictionary.get('radio-check_protocol')[0]
                casevac_protocol_complete = sequence_dictionary.get('casevac_protocol_complete')[0]
                casevac_protocol_variation_1 = sequence_dictionary.get('casevac_protocol_variation_1')[0]
                casevac_protocol_variation_2 = sequence_dictionary.get('casevac_protocol_variation_2')[0]
                sitrep_protocol_complete = sequence_dictionary.get('sitrep_protocol_complete')[0]
                sitrep_protocol_variation_1 = sequence_dictionary.get('sitrep_protocol_variation_1')[0]
                sitrep_protocol_variation_2 = sequence_dictionary.get('sitrep_protocol_variation_2')[0]
                locstat_protocol_complete = sequence_dictionary.get('locstat_protocol_complete')[0]
                locstat_protocol_variation_1 = sequence_dictionary.get('locstat_protocol_variation_1')[0]
                locstat_protocol_variation_2 = sequence_dictionary.get('locstat_protocol_variation_2')[0]
                atmist_protocol_complete = sequence_dictionary.get('atmist_protocol_complete')[0]
                atmist_protocol_variation_1 = sequence_dictionary.get('atmist_protocol_variation_1')[0]
                atmist_protocol_variation_2 = sequence_dictionary.get('atmist_protocol_variation_2')[0]
                opdem_protocol_complete = sequence_dictionary.get('opdem_protocol_complete')[0]
                opdem_protocol_variation_1 = sequence_dictionary.get('opdem_protocol_variation_1')[0]
                opdem_protocol_variation_2 = sequence_dictionary.get('opdem_protocol_variation_2')[0]

                # calculate the sequence scoring against each of the message sequence protocols
                self.nlp_dataframe_all['radio-check_protocol_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=radio_check_protocol).ratio())
                self.nlp_dataframe_all['casevac_protocol_complete_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=casevac_protocol_complete).ratio())
                self.nlp_dataframe_all['casevac_protocol_var1_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=casevac_protocol_variation_1).ratio())
                self.nlp_dataframe_all['casevac_protocol_var2_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=casevac_protocol_variation_2).ratio())
                self.nlp_dataframe_all['sitrep_protocol_complete_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=sitrep_protocol_complete).ratio())
                self.nlp_dataframe_all['sitrep_protocol_var1_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=sitrep_protocol_variation_1).ratio())
                self.nlp_dataframe_all['sitrep_protocol_var2_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=sitrep_protocol_variation_2).ratio())
                self.nlp_dataframe_all['locstat_protocol_complete_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=locstat_protocol_complete).ratio())
                self.nlp_dataframe_all['locstat_protocol_var1_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=locstat_protocol_variation_1).ratio())
                self.nlp_dataframe_all['locstat_protocol_var2_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=locstat_protocol_variation_2).ratio())
                self.nlp_dataframe_all['atmist_protocol_complete_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=atmist_protocol_complete).ratio())
                self.nlp_dataframe_all['atmist_protocol_var1_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=atmist_protocol_variation_1).ratio())
                self.nlp_dataframe_all['atmist_protocol_var2_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=atmist_protocol_variation_2).ratio())
                self.nlp_dataframe_all['opdem_protocol_complete_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=opdem_protocol_complete).ratio())
                self.nlp_dataframe_all['opdem_protocol_var1_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=opdem_protocol_variation_1).ratio())
                self.nlp_dataframe_all['opdem_protocol_var2_score'] = self.nlp_dataframe_all['sequence_hashed'].apply(lambda x: SequenceMatcher(a=x , b=opdem_protocol_variation_2).ratio())


                # also perform the maximum and minimuma across all these scrores
                similarity_score_column_list = ['radio-check_protocol_score', 'casevac_protocol_complete_score', 'casevac_protocol_var1_score', 'casevac_protocol_var2_score', 'sitrep_protocol_complete_score',
                                'sitrep_protocol_var1_score', 'sitrep_protocol_var2_score', 'locstat_protocol_complete_score', 'locstat_protocol_var1_score',
                                'locstat_protocol_var2_score', 'atmist_protocol_complete_score', 'atmist_protocol_var1_score','atmist_protocol_var2_score',
                                'opdem_protocol_complete_score', 'opdem_protocol_var1_score', 'opdem_protocol_var2_score']
                
                # get maximum and minimums across rows
                self.nlp_dataframe_all['sequence_score_max'] = self.nlp_dataframe_all[similarity_score_column_list].max(axis=1)
                self.nlp_dataframe_all['sequence_score_min'] = self.nlp_dataframe_all[similarity_score_column_list].min(axis=1)

                # get the column (score) with this max value
                self.nlp_dataframe_all['sequence_match'] = self.nlp_dataframe_all[similarity_score_column_list].idxmax(axis=1)
                self.nlp_dataframe_all['sequence_match_protocol'] = self.nlp_dataframe_all['sequence_match'].apply(lambda x: x[:x.find('_')]).astype(str)

            else:
                raise Exception('Protocol sequence dictionary does not exist')  

        except Exception as error:
            print(f'Error in {error}') 


    def saveNLPResults(self, datastore, file_path, target_path, transcript_dataframe, transcript_type='single'):
        """
        Function to save the results of the NLP modelliong proceses which includes
        key phrase extraction, NER and topic modelling.
        """
        try:
            if (datastore is not None):
                
                # Save the NLP (global) as Serialize data into file:
                if len(self.nlp_dict_all) > 0:
                    json.dump(self.nlp_dict_all, open(f'{file_path}NLP_dict.json', 'w' ))
                else:
                    raise Exception('Error in nlp dataframe')

                # save the disctionary also as dataframe for visualisation & serving purposes
                if self.nlp_dataframe_all is not None:
                    self.nlp_dataframe_all.to_csv(f'{file_path}NLP_dataframe.csv', sep =',', header=True, index=False)

        
                # also save in the registred datastore (reflected in the Azure Blob)
                results_list = []
                for path, subdirs, files in os.walk(file_path):
                    for name in files:
                        results_list.append(os.path.join(path, name))

                if(datastore is not None):
                    #upload into datastore
                    datastore.upload_files(files = results_list, relative_root=file_path, 
                                            target_path= target_path, overwrite=True)   
                    print(f'Upload NLP results to wrting results for {datastore.name}')
                else:
                    #raise Exception('Error in uploading results to datastore')
                    print('No Datastore provide to upload results')

            else:
                raise Exception('Error in datastore')
                    
        except Exception as error:
            print(f'Error in {error}') 

