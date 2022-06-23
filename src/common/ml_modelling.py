# Script to perform the ML modelling based on the prepared dataframes from the NLP processing

# setup the current paths
import os, sys
currentDir = os.path.dirname(os.getcwd())
sys.path.append(currentDir)

# system related
import pandas as pd
import numpy as np
import pickle

# ML libraries - Sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
#import autosklearn.classification

from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import ScriptRunConfig

import matplotlib.pyplot as plt
import seaborn as sns

# import from common setups & environments
from common.constants import *

class MLModelling():
    """
    This class focuses on the prediction modelling 
    """
    def __init__(self):
        super().__init__() # inherit if applicable
        
        self.ml_dataframe_all = pd.DataFrame()
        self.ml_dataframe_reduced = pd.DataFrame()
        self.ml_datraframe_independents = pd.DataFrame()
        self.ml_datraframe_dependents = pd.DataFrame()
        self.ml_dataframe_train = pd.DataFrame()
        self.ml_dataframe_test = pd.DataFrame()
        self.ml_dataframe_inference = pd.DataFrame()
        self.dataframe_all = pd.DataFrame()
        self.inference_dataframe_all = pd.DataFrame()
        self.inference_dataframe_compact = pd.DataFrame()

        self.__min_max_scaler = None
        self.script_run_config = None
        self.y_predicted_test = None
        self.y_predict_test_prob = None
        self.y_predicted_train = None
        self.y_predict_train_prob = None
        self.y_inference = None
        self.y_prob = list()

        # heuristic approach
        self.heuristic_predicted = list()
        self.heuristic_predicted_num = list()

        self.tfidf_vectors = None


    def configMLDataframe(self, dataframe, normalize=False):
        """
        Function to configure the ML dataframe
        """
        try:

            if dataframe is not None:
                # remove null rows (based on text column)
                dataframe.dropna(subset=['text'], inplace=True)
                
                # the columns to drop
                drop_column_list = ['start_date', 
                                    'start_time', 
                                    'end_date', 
                                    'end_time', 
                                    'display_text', 
                                    'language',
                                    'confidence', 
                                    'offset',
                                    'filtered_tokenized_transcript', 
                                    'token_fdisk', 
                                    'keyPhrases', 
                                    'warnings', 
                                    'general_phrases', 
                                    'entities'
                                    ]
                
                # assign the ml dataframe (complete)
                self.ml_dataframe_all = dataframe.drop(drop_column_list, axis=1)

                # apply numeric values to labels
                if 'ground_truth_class' in self.ml_dataframe_all.columns:
                   self.ml_dataframe_all['ground_truth_class_num'] = self.ml_dataframe_all['ground_truth_class'].apply(lambda x: 0 if x == MESSAGE_CLASSIFICATION_GROUP[0] else 
                                                (1 if x == MESSAGE_CLASSIFICATION_GROUP[1] else
                                                (2 if x == MESSAGE_CLASSIFICATION_GROUP[2] else  0))).astype(np.int8)

                # convert the duration and offsets to seconds
                self.ml_dataframe_all['duration'] = (self.ml_dataframe_all['duration']/1000).astype(float)

                # check to see if require normalisation
                if normalize:
                    # setup min max scaler (may be required)
                    self.__min_max_scaler = MinMaxScaler(feature_range=(0, 1))
                    
                    # word count
                    temp_array = np.array(self.ml_dataframe_all['word_count']).reshape(-1,1)
                    self.ml_dataframe_all['word_count'] = self.__min_max_scaler.fit_transform(temp_array)

                    # number of tokens found
                    temp_array = np.array(self.ml_dataframe_all['num_filtered_tokens']).reshape(-1,1)
                    self.ml_dataframe_all['num_filtered_tokens'] = self.__min_max_scaler.fit_transform(temp_array)
                
            else:
                raise Exception('dataframe does not exist')    

        except Exception as error:
            print(f'Error in {error}')    

    
    def prepareMLDataframe(self, train):
        """
        Function to further reduce the ML dataframe for modelling
        """
        try:
            # the columns to drop
            keep_column_list = ML_KEEP_COLUMNS

            # if for training keep the truth column
            if self.ml_dataframe_all is not None:
                
                if train:
                    keep_column_list.append('ground_truth_class_num')    
                    
                    # create an ML dataframe
                    self.ml_dataframe_reduced = self.ml_dataframe_all[keep_column_list]

                    # assign the dependant & independant variables
                    self.ml_datraframe_independents = self.ml_dataframe_reduced.drop(columns='ground_truth_class_num', axis=1)
                    self.ml_datraframe_dependents = self.ml_dataframe_reduced['ground_truth_class_num']

                # inference
                else:
                    self.ml_dataframe_inference = self.ml_dataframe_all[keep_column_list]

            else:
                raise Exception('dataframe does not exist')    

        except Exception as error:
            print(f'Error in {error}') 

    def exploratoryDataAnalysis(self, dataframe, plot=True):
        """
        Function to perform a series of exploratory data analysis (EDA) on
        the dataframe
        """

        try:
            if plot:
                print('Ploting data for analysis')
                COLOR = 'white'
                plt.rcParams['text.color'] = COLOR
                plt.rcParams['axes.labelcolor'] = COLOR
                plt.rcParams['xtick.color'] = COLOR
                plt.rcParams['ytick.color'] = COLOR

                # interactions
                scenario_group = dataframe.groupby(['ground_truth_class'])['ground_truth_class'].count()
                x = scenario_group.index

                # word count
                word_count_group = dataframe.groupby(['ground_truth_class'])['word_count'].mean()

                #duration
                duration_group = dataframe.groupby(['ground_truth_class'])['message_duration'].mean()

                # call signs found
                call_signs_found_group = dataframe.groupby(['ground_truth_class'])['call_signs_found_n'].mean()

                # similarities
                casevac_protocol_complete = dataframe.groupby(['ground_truth_class'])['casevac_protocol_complete_score'].mean()
                sitrep_protocol_complete = dataframe.groupby(['ground_truth_class'])['sitrep_protocol_complete_score'].mean()
                locstat_protocol_complete = dataframe.groupby(['ground_truth_class'])['locstat_protocol_complete_score'].mean()
                atmist_protocol_complete = dataframe.groupby(['ground_truth_class'])['atmist_protocol_complete_score'].mean()
                    

                # plot 1
                #plt.subplot(2, 3, 1)
                fig, ax = plt.subplots(figsize=(12,9))
                plt.bar(x = x,  height=word_count_group)
                plt.xlabel('Scenario')
                plt.title('Word count (mean)')
                plt.show()

                # plot 2
                #plt.subplot(2, 3, 2)
                fig, ax = plt.subplots(figsize=(12,9))
                plt.bar(x = x,  height=duration_group)
                plt.xlabel('Scenario')
                plt.title('duration_group (mean)')
                plt.show()

                # plot 3
                #plt.subplot(2, 3, 3)
                fig, ax = plt.subplots(figsize=(12,9))
                plt.bar(x = x,  height=call_signs_found_group)
                plt.xlabel('Scenario')
                plt.title('call signs found (mean)')
                plt.show()

                # plot 4
                #plt.subplot(2, 3, 4)
                fig, ax = plt.subplots(figsize=(12,9))
                plt.bar(x = x,  height=casevac_protocol_complete)
                plt.xlabel('Scenario')
                plt.title('Casevac complete score')
                plt.show()


                #plt.subplot(2, 1, 1)
                fig, ax = plt.subplots(figsize=(12,9))
                plt.scatter(dataframe['casevac_protocol_complete_score'],dataframe['sitrep_protocol_complete_score'] )
                plt.xlabel('casevac_protocol_complete_score')
                plt.ylabel('sitrep_protocol_complete_score')
                plt.title('Casevac vs sitrep score ')
                plt.show()

                # correlation analysis - on message protocol truths
                fig, ax = plt.subplots(figsize=(12,9))
                corr_df = dataframe[['casevac_protocol_complete_score', 'sitrep_protocol_complete_score', 
                                    'locstat_protocol_complete_score', 'atmist_protocol_complete_score']]

                hm = sns.heatmap(corr_df.corr(), annot = True)
                hm.set(xlabel='\nProtocols', ylabel='protocols\t', title = 'Protocol scores\n')
                plt.show()


                print(corr_df.corr())

        except Exception as error:
            print(f'Error in {error}') 


    def configScriptRun(self, arg_list, script_dir, script, compute_target, environment):
        """
        Function to configure the script to submit for Azure ML
        """     
        try:
            self.script_run_config = ScriptRunConfig(source_directory=script_dir,
                                script=script, 
                                arguments=arg_list,
                                compute_target=compute_target,
                                environment=environment)

        except Exception as error:
            print(f'Error in {error}')

    def prepareTrainTestSplit(self, test_size=0.33):
        """
        Function to split the ml dataframe into trainng and testing
        """
        try:
            if (self.ml_datraframe_independents is not None and self.ml_datraframe_dependents is not None):
                X = self.ml_datraframe_independents
                y = self.ml_datraframe_dependents

                X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)

                self.ml_datraframe_independents_train = X_train
                self.ml_datraframe_independents_test = X_test
                
                self.ml_datraframe_dependents_train = y_train
                self.ml_datraframe_dependents_test = y_test

                #also keep the train and test dataframes
                # training set all
                self.ml_dataframe_train = pd.concat([self.ml_datraframe_independents_train, self.ml_datraframe_dependents_train], 
                                                    axis=1, ignore_index=False)
                # test set all
                self.ml_dataframe_test = pd.concat([self.ml_datraframe_independents_test, self.ml_datraframe_dependents_test], 
                                                    axis=1, ignore_index=False)
                
            else:
                raise Exception ('Dataframe is empty')
               
        except Exception as error:
            print(f'Error in {error}')    

    def tfidfVectorizer(self, dataframe):
        """
        Function to create the TF-IDF vectorizer, to be used
        for scenario classification
        """
        try:
            if len(dataframe) > 0:
                # setup the required training vector    
                X_train = dataframe['text']

                # setup TFIDF vector
                tfidf_vectorizer = TfidfVectorizer(use_idf=True)
                self.tfidf_vectors = tfidf_vectorizer.fit_transform(X_train)

            else:
                raise Exception('Dataframe is empty')

        except Exception as error:
            print(f'Error in {error}') 


    def __fitModel(self, model):
        """
        Function (private) to prcocess the given model
        """
        try:
            results_list = list()
            y_predict_prob = list()
            y_predict_test_prob = list()
            y_predict_train_prob = list()

            # fit the given model
            if model is not None:
            
                # fit the model to the training data
                if self.ml_datraframe_independents_train is not None and self.ml_datraframe_dependents_train is not None:
                    model.fit(X=self.ml_datraframe_independents_train, y=self.ml_datraframe_dependents_train)
                
                    # perform the prediction
                    # here we will apply to both training and test (for completion)
                    
                    # test set
                    y_predict_test = model.predict(X=self.ml_datraframe_independents_test)
                    y_predict_prob = model.predict_proba(X=self.ml_datraframe_independents_test)
                    # get only probability for positive case
                    for prob_list in y_predict_prob:
                        y_predict_test_prob.append(prob_list[1])

                    # train
                    y_predict_train = model.predict(X=self.ml_datraframe_independents_train)
                    y_predict_prob = model.predict_proba(X=self.ml_datraframe_independents_train)
                    # get only probability for positive case
                    for prob_list in y_predict_prob:
                        y_predict_train_prob.append(prob_list[1])
                    
                    # analyse the performance
                    y_true = self.ml_datraframe_dependents_test

                    cm = confusion_matrix(y_true=y_true, y_pred=y_predict_test)
                    TN, FP, FN, TP = cm.ravel()

                    accuracy = accuracy_score(y_true=y_true, y_pred=y_predict_test)
                    precision = precision_score(y_true=y_true, y_pred=y_predict_test, average='binary')
                    recall = recall_score(y_true=y_true, y_pred=y_predict_test, average='binary')   

                    # prepare the results
                    results_list.extend([y_predict_test, y_predict_test_prob, y_predict_train, y_predict_train_prob, cm, accuracy, precision, recall]) 
                else:
                    raise Exception ('Training dataframe is empty')    
            else:
                raise Exception ('Model is empty')  

        except Exception as error:
            print(f'Error in {error}') 
        finally:
            return results_list

    def __plotModel(self, model, cm):
        """
        Function (private) to plot the confusion matrix
        """
        try:
            # prepare the confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot()
            plt.show()

        except Exception as error:
            print(f'Error in {error}') 

    def __saveModel(self, model, model_path):
        """
        Function (private) to save the model as pickle file
        """
        try:
           # save model
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)

        except Exception as error:
            print(f'Error in {error}') 

    def logisticRegression(self, model_path, plot=True):
        """
        Function to perform baseline logistic regression
        """
        try:
            results_list = list()
            # initialise the model
            logit = LogisticRegression(penalty='l2', tol=0.0001, C=1.0/0.4, solver="liblinear", max_iter=200, 
                            multi_class='auto', verbose=0)
            
            results_list = self.__fitModel(logit)
            print(f'Accuracy: {results_list[5]}')
            print(f'Precision: {results_list[6]}')
            print(f'Recall: {results_list[7]}')

            # plot the confusion matrix if required 
            if plot:
                self.__plotModel(logit, results_list[4])
                    
            # save model        
            self.__saveModel(model=logit, model_path=model_path)    
            
            # save the predicted values
            self.y_predicted_test = results_list[0]
            self.y_predict_test_prob = results_list[1]
            self.y_predicted_train = results_list[2]
            self.y_predict_train_prob = results_list[3] 

        except Exception as error:
            print(f'Error in {error}') 

    def randomForest(self, model_path, plot=True):
        """
        Function to perform random Forest classifier
        """
        try:
            results_list = list()
            # initialise the model
            randomforest = RandomForestClassifier(criterion='gini', max_features='auto', min_impurity_decrease=0.0, 
                                    bootstrap=True, warm_start=False, class_weight=None, ccp_alpha=0.0, 
                                    max_samples=None)
            
                      
            results_list = self.__fitModel(randomforest)
            print(f'Accuracy: {results_list[5]}')
            print(f'Precision: {results_list[6]}')
            print(f'Recall: {results_list[7]}')

            # plot the confusion matrix if required 
            if plot:
                self.__plotModel(randomforest, results_list[4])
                    
            # save model        
            self.__saveModel(model=randomforest, model_path=model_path)    
            
            # save the predicted values
            self.y_predicted_test = results_list[0]
            self.y_predict_test_prob = results_list[1]
            self.y_predicted_train = results_list[2]
            self.y_predict_train_prob = results_list[3] 
            
        except Exception as error:
            print(f'Error in {error}') 

    
    
    def randomForest_special(self, model_path, plot=True):
        """
        Function to perform random Forest classifier on entire set
        """
        try:
            # initialise the model
            randomforest = RandomForestClassifier(criterion='gini', max_features='auto', min_impurity_decrease=0.0, 
                                    bootstrap=True, warm_start=False, class_weight=None, ccp_alpha=0.0, 
                                    max_samples=None)          
            
            # fit the model
            y_predict_prob = list()

            # fit the given model
            if randomforest is not None:
            
                # fit the model to the training data
                if self.ml_datraframe_independents is not None and self.ml_datraframe_dependents is not None:
                    randomforest.fit(X=self.ml_datraframe_independents, y=self.ml_datraframe_dependents)
                
                    # perform the prediction - in this case on entire set agaian
                    y_predict = randomforest.predict(X=self.ml_datraframe_independents)
                    y_prob = randomforest.predict_proba(X=self.ml_datraframe_independents)
                    # get only probability for positive case
                    for prob_list in y_prob:
                        y_predict_prob.append(prob_list[1])
                    
                    # analyse the performance
                    y_true = self.ml_datraframe_dependents

                    cm = confusion_matrix(y_true=y_true, y_pred=y_predict)

                    accuracy = accuracy_score(y_true=y_true, y_pred=y_predict)
                    precision = precision_score(y_true=y_true, y_pred=y_predict, average='macro')
                    recall = recall_score(y_true=y_true, y_pred=y_predict, average='macro')   

                    print('Random Forest Model performance')
                    print(f'Accuracy: {accuracy}')
                    print(f'Precision: {precision}')
                    print(f'Recall: {recall}')

                    # plot the confusion matrix if required 
                    if plot:
                        self.__plotModel(randomforest, cm)
                                        
        except Exception as error:
            print(f'Error in {error}')
    
    
    def gradientBoosting(self, model_path, plot=True):
        """
        Function to perform gradinet boosting classifier
        """
        try:
            results_list = list()
            # initialise the model
            
            gradientBoost = GradientBoostingClassifier(criterion= 'friedman_mse',learning_rate=0.1, n_estimators=100,  
                                        max_depth=3, min_impurity_decrease=0.0, validation_fraction=0.1, 
                                        n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
            
            
            results_list = self.__fitModel(gradientBoost)
            print(f'Accuracy: {results_list[5]}')
            print(f'Precision: {results_list[6]}')
            print(f'Recall: {results_list[7]}')

            # plot the confusion matrix if required 
            if plot:
                self.__plotModel(gradientBoost, results_list[4])
                    
            # save model        
            self.__saveModel(model=gradientBoost, model_path=model_path)    
            
            # save the predicted values
            self.y_predicted_test = results_list[0]
            self.y_predict_test_prob = results_list[1]
            self.y_predicted_train = results_list[2]
            self.y_predict_train_prob = results_list[3] 
            
        except Exception as error:
            print(f'Error in {error}') 

    def autoML(self, model_path, plot=True):
        """
        Function to perform auto ML classifier
        """
        try:
            results_list = list()
            
            # initialise the model
            autoMLClassifier = autosklearn.classification.AutoSklearnClassifier()
            
            results_list = self.__fitModel(autoMLClassifier)
            print(f'Accuracy: {results_list[3]}')
            print(f'Precision: {results_list[4]}')
            print(f'Recall: {results_list[5]}')

            # plot the confusion matrix if required 
            if plot:
                self.__plotModel(autoMLClassifier, results_list[2])
                    
            # save model        
            self.__saveModel(model=autoMLClassifier, model_path=model_path)    
            
            # save the predicted values
            self.y_predicted_test = results_list[0]
            self.y_predicted_train = results_list[1]

        except Exception as error:
            print(f'Error in {error}') 

    # def heuristicApproach(self, dataframe):
    #     """
    #     Heuritstic (rule-base) approach towards inferenceing the message classification
    #     """
    #     try:
    #         if len(dataframe) > 0:
    #             # process the rules each row at a time - can also use lambda functions
    #             for row in range(len(dataframe)):
    #                 # obtain the features
    #                 tokens = dataframe.loc[row,'num_filtered_tokens'] 
    #                 scenario_primary = dataframe.loc[row,'scenario_primary']
    #                 casevac_stops = dataframe.loc[row,'casevac_stops_found_n']
    #                 time_found = dataframe.loc[row,'time_found']
    #                 grid_found = dataframe.loc[row,'grid_found']
    #                 sequence_match_protocol = dataframe.loc[row,'sequence_match_protocol']
    #                 vehicle_phrases_list = dataframe.loc[row,'vehicle_phrases']
    #                 kinetic_phrases_list = dataframe.loc[row,'kinetic_phrases']


    #                 # rule 1 - radio-check
    #                 # --------------------
    #                 if (tokens <= TOKEN_COUNT_CUTOFF_LOWEST) or (scenario_primary == MESSAGE_CLASSIFICATION_GROUP[1].lower()):
    #                     self.heuristic_predicted.append(MESSAGE_CLASSIFICATION_GROUP[1])

    #                 # rule 2 - casevac
    #                 # ----------------
    #                 elif casevac_stops >= CASEVAC_STOPS_CUTTOFF:
    #                     self.heuristic_predicted.append(MESSAGE_CLASSIFICATION_GROUP[2])  

    #                 # rule 3 - locstat
    #                 # ----------------
    #                 elif (TOKEN_COUNT_CUTOFF_LOWEST < tokens <= TOKEN_COUNT_CUTOFF_MIDDLE) and \
    #                     (scenario_primary == MESSAGE_CLASSIFICATION_GROUP[4].lower() or sequence_match_protocol == MESSAGE_CLASSIFICATION_GROUP[4].lower()) and \
    #                     (len(vehicle_phrases_list) == 0 and len(kinetic_phrases_list) == 0):
    #                     self.heuristic_predicted.append(MESSAGE_CLASSIFICATION_GROUP[4])

    #                 # rule 4 - sitrep
    #                 # ---------------
    #                 elif (TOKEN_COUNT_CUTOFF_LOWEST < tokens <= TOKEN_COUNT_CUTOFF_UPPER) and (time_found == 1 or grid_found == 1 or sequence_match_protocol == MESSAGE_CLASSIFICATION_GROUP[5]):
    #                     self.heuristic_predicted.append(MESSAGE_CLASSIFICATION_GROUP[5])

    #                 # rule 5 - atmist
    #                 # ---------------
    #                 elif tokens > TOKEN_COUNT_CUTOFF_UPPER and scenario_primary == MESSAGE_CLASSIFICATION_GROUP[3].lower():
    #                     self.heuristic_predicted.append(MESSAGE_CLASSIFICATION_GROUP[3])

    #                 # rule 6 - other
    #                 else:
    #                     self.heuristic_predicted.append(MESSAGE_CLASSIFICATION_GROUP[0])


    #             # aqlso process the encoded version
    #             self.heuristic_predicted_num = list(map(lambda x: 1 if x == MESSAGE_CLASSIFICATION_GROUP[1] else 
    #                                                     (2 if x == MESSAGE_CLASSIFICATION_GROUP[2] else
    #                                                     (3 if x == MESSAGE_CLASSIFICATION_GROUP[3] else 
    #                                                     (4 if x == MESSAGE_CLASSIFICATION_GROUP[4] else 
    #                                                     (5 if x == MESSAGE_CLASSIFICATION_GROUP[5] else 0)))), self.heuristic_predicted))

    #             # analyse the performance
    #             y_true = self.ml_datraframe_dependents
    #             y_pred = self.heuristic_predicted_num

    #             cm = confusion_matrix(y_true = y_true, y_pred = y_pred)
    #             accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    #             precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    #             recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')  
                
    #             print('Heuristic Rule-base performance')
    #             print(f'Accuracy: {accuracy}')
    #             print(f'Precision: {precision}')
    #             print(f'Recall: {recall}')

    #         else:
    #             raise Exception('Dataframe for applying rule base is empty')    

    #     except Exception as error:
    #         print(f'Error in {error}') 


    def processMLResults(self, predicted_list, dataframe):
        """
        Function to process the ML results and process the final dataframe
        """
        try:
            
            # merge with the main NLP dataframe
            if len(dataframe) > 0:
                dataframe['heuristic_predicted'] = predicted_list      
            
            else:
                raise Exception('Dataframe is empty')
            
        except Exception as error:
            print(f'Error in {error}')


    def processMLResults_old(self, predicted_list_test, predicted_list_test_p, predicted_list_train, predicted_list_train_p, dataframe):
        """
        Function to process the ML results and process the final dataframe
        """
        try:
            column_list = ['interaction_predicted_test', 'interaction_predicted_test_p', 'interaction_predicted_train', 
            'interaction_predicted_train_p', 'test', 'train']

            # merge with the main ML dataframe
            # perform for the test
            if self.ml_dataframe_test is not None:
                self.ml_dataframe_test['interaction_predicted'] = predicted_list_test
                self.ml_dataframe_test['interaction_predicted_prob'] = predicted_list_test_p
            
            if self.ml_dataframe_test is not None:
                self.ml_dataframe_train['interaction_predicted'] = predicted_list_train
                self.ml_dataframe_train['interaction_predicted_prob'] = predicted_list_train_p

            # process the predicted results from both test and train scoring
            temp_test_df = pd.DataFrame(self.ml_dataframe_test['interaction_predicted'])
            temp_test_df['interaction_predicted_test'] = temp_test_df['interaction_predicted'].apply(lambda x: 'y' if x == 1 else 'n').astype(str)
            temp_test_df['test'] = 'test'
            temp_test_df['interaction_predicted_test_p'] = pd.DataFrame(self.ml_dataframe_test['interaction_predicted_prob']).astype(str)

            temp_train_df = pd.DataFrame(self.ml_dataframe_train['interaction_predicted'])
            temp_train_df['interaction_predicted_train'] = temp_train_df['interaction_predicted'].apply(lambda x: 'y' if x == 1 else 'n').astype(str)
            temp_train_df['train'] = 'train'
            temp_train_df['interaction_predicted_train_p'] = pd.DataFrame(self.ml_dataframe_train['interaction_predicted_prob']).astype(str)


            # merge into the main dataframe
            self.dataframe_all = pd.concat([dataframe, temp_test_df, temp_train_df], axis=1, ignore_index=False)
            self.dataframe_all[column_list] = self.dataframe_all[column_list].fillna('')
            self.dataframe_all.drop(['interaction_predicted'], axis=1, inplace=True)

            # combine some of the columns
            self.dataframe_all['train_test'] = self.dataframe_all[['test', 'train']].agg(''.join, axis=1)
            self.dataframe_all['interaction_predicted'] = self.dataframe_all[['interaction_predicted_test', 'interaction_predicted_train']].agg(''.join, axis=1)
            self.dataframe_all['interaction_predicted_prob'] = self.dataframe_all[['interaction_predicted_test_p', 'interaction_predicted_train_p']].sum(axis=1)
            
            self.dataframe_all.drop(column_list, axis=1, inplace=True)

            
        except Exception as error:
            print(f'Error in {error}')


    def boostModel(self):
        """
        Function to boost the ML prediction by applying heuristic rules
        """
        try:
            
            if self.dataframe_all is not None:
                
                # rule 1: duration
                #-----------------
                DURATION_THRESHOLD = 1
                self.dataframe_all['interaction_predicted_boosted'] = self.dataframe_all[['interaction_predicted', 'duration_metadata']].apply(lambda x: 'n' if x['duration_metadata'] < DURATION_THRESHOLD else x['interaction_predicted'], axis=1).astype(str)

                # rule 2: meta description 
                #-------------------------
                DESCRIPTION_LIST = ['MAC Mine 6', 'MAC Mine West - 49B', 'MAC Supervisor - West']
                self.dataframe_all['interaction_predicted_boosted'] = self.dataframe_all[['interaction_predicted_boosted', 'From_descr']].apply(lambda x: 'n' if x['From_descr'] in DESCRIPTION_LIST else x['interaction_predicted_boosted'], axis=1).astype(str)

                # rule 3: duration
                #-----------------
                WORD_COUNT_THRESHOLD = 2
                self.dataframe_all['interaction_predicted_boosted'] = self.dataframe_all[['interaction_predicted_boosted', 'word_count']].apply(lambda x: 'n' if x['word_count'] < WORD_COUNT_THRESHOLD else x['interaction_predicted_boosted'], axis=1).astype(str)

                # format the dataframe for post analysis
                self.dataframe_all['interaction_predicted_num'] = self.dataframe_all['interaction_predicted'].apply(lambda x: 1 if x == 'y' else 0).astype(int)
                self.dataframe_all['interaction_truth_class_num'] = self.dataframe_all['interaction_truth_class'].apply(lambda x: 1 if x == 'y' else 0).astype(int)
                self.dataframe_all['interaction_predicted_boosted_num'] = self.dataframe_all['interaction_predicted_boosted'].apply(lambda x: 1 if x == 'y' else 0).astype(int)


                y_true = self.dataframe_all['interaction_truth_class_num']
                y_pred = self.dataframe_all['interaction_predicted_num']
                y_pred = self.dataframe_all['interaction_predicted_boosted_num']

                accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
                precision = precision_score(y_true=y_true, y_pred=y_pred, average='binary')
                recall = recall_score(y_true=y_true, y_pred=y_pred, average='binary') 
                print('Post modelling analysis')
                print(accuracy,precision,recall)

                # cleanup
                self.dataframe_all.drop(['interaction_truth_class_num', 'interaction_predicted_num', 'interaction_predicted_boosted_num'],axis=1, inplace=True)
     
            else:
                raise Exception ('ML dataframe is empty')


        except Exception as error:
            print(f'Error in {error}') 
    
    def submitModel(self):
        """
        Function to submit the model run to the ML services
        """
        try:

            # setup the training script run configuration
            print('test')

        except Exception as error:
            print(f'Error in {error}') 


    def inferencing(self, model_path, dataframe):
        """
        Function to perform the inferenceing of given dataframe and
        given best model
        """
        try:
            if dataframe is not None:

                # extract the best model
                best_model = pickle.load(open(model_path, 'rb'))

                # perform the inferencing
                self.y_inference = best_model.predict(X=dataframe)
                y_predict_prob = best_model.predict_proba(X=dataframe)

                # get only probability for positive case
                for prob_list in y_predict_prob:
                    self.y_prob.append(prob_list[1])
            
            else:
                raise Exception ('The inference Dataframe is empty')

        except Exception as error:
            print(f'Error in {error}') 

    def processInferenceResults(self, predicted_list, predicted_list_p, dataframe):
        """
        Function to process the inferenced results and process the final dataframe
        """
        try:

            column_list = ['interaction_predicted', 'interaction_predicted_prob']

            # merge with the main ML dataframe
            # perform for the inference
            if self.ml_dataframe_inference is not None:
                self.ml_dataframe_inference['interaction_predicted'] = predicted_list
                self.ml_dataframe_inference['interaction_predicted_prob'] = predicted_list_p

                # process in temp dataframe
                temp_inference_df = pd.DataFrame(self.ml_dataframe_inference['interaction_predicted'])
                temp_inference_df['interaction_predicted'] = temp_inference_df['interaction_predicted'].apply(lambda x: 'y' if x == 1 else 'n').astype(str)
                temp_inference_df['interaction_predicted_prob'] = pd.DataFrame(self.ml_dataframe_inference['interaction_predicted_prob']).astype(str)

                # merge into the main dataframe
                self.inference_dataframe_all = pd.concat([dataframe, temp_inference_df], axis=1, ignore_index=False)

                self.inference_dataframe_all[column_list] = self.inference_dataframe_all[column_list].fillna('')

        except Exception as error:
            print(f'Error in {error}')  
            
    def boostInference(self):
        """
        Function to boost the ML prediction by applying heuristic rules
        """
        try:
            
            if self.inference_dataframe_all is not None:
                
                # rule 1: duration
                #-----------------
                DURATION_THRESHOLD = 1
                self.inference_dataframe_all['interaction_predicted_boosted'] = self.inference_dataframe_all[['interaction_predicted', 'duration_metadata']].apply(lambda x: 'n' if x['duration_metadata'] < DURATION_THRESHOLD else x['interaction_predicted'], axis=1).astype(str)

                # rule 2: meta description 
                #-------------------------
                DESCRIPTION_LIST = ['MAC Mine 6', 'MAC Mine West - 49B', 'MAC Supervisor - West']
                self.inference_dataframe_all['interaction_predicted_boosted'] = self.inference_dataframe_all[['interaction_predicted_boosted', 'From_descr']].apply(lambda x: 'n' if x['From_descr'] in DESCRIPTION_LIST else x['interaction_predicted_boosted'], axis=1).astype(str)

                # rule 3: duration
                #-----------------
                WORD_COUNT_THRESHOLD = 2
                self.inference_dataframe_all['interaction_predicted_boosted'] = self.inference_dataframe_all[['interaction_predicted_boosted', 'word_count']].apply(lambda x: 'n' if x['word_count'] < WORD_COUNT_THRESHOLD else x['interaction_predicted_boosted'], axis=1).astype(str)
     
            else:
                raise Exception ('ML dataframe is empty')
                
        except Exception as error:
            print(f'Error in {error}') 


    def saveMLResults(self, datastore, file_path, ml_file_name, all_file_name, target_path, ml_dataframe, all_dataframe):
        """
        Function to save the results of the Ml modelling 
        """
        try:
            # save the ML dataframe
            if ml_dataframe is not None:
                ml_dataframe.to_csv(f'{file_path}{ml_file_name}', sep =',', header=True, index=True)
            
            # save the ML (all) dataframe
            if all_dataframe is not None:
                all_dataframe.to_csv(f'{file_path}{all_file_name}', sep =',', header=True, index=True)
            
            # also store in datastore
            if(datastore is not None):             
                   
                # also save in the registred datastore (reflected in the Azure Blob)
                results_list = []
                for path, subdirs, files in os.walk(file_path):
                    for name in files:
                        results_list.append(os.path.join(path, name))

                #upload into datastore
                datastore.upload_files(files = results_list, relative_root=file_path, 
                                        target_path= target_path, overwrite=True)   
                print(f'Upload ML modelling results to wrting results for {datastore.name}')
                
            else:
                raise Exception('Error in uploading results to datastore')
      
        except Exception as error:
            print(f'Error in {error}') 

    def saveInferenceResults(self, datastore, file_path, all_file_name, compact_file_name, target_path, all_dataframe):
        """
        Function to save the results of the inference 
        """
        try:
            drop_columns = list()
            dataframe_compact = pd.DataFrame()

            # save the inferenced dataframe
            if all_dataframe is not None:
                all_dataframe.to_csv(f'{file_path}{all_file_name}', sep =',', header=True, index=True)
            
                # drop some columns to also save the compact version of the dataframe
                col_names = list(all_dataframe.columns)

                # loop through and extract the columns to drop
                for col in col_names:
                    if (('asset_' in col)) and ((col != 'asset_list') and (col != 'num_asset_found') and (col != 'asset_found')):
                        drop_columns.append(col)
                    
                    if (('key_phrase_' in col) or ('position_' in col) or ('other_keyword_' in col) or ('vehicle_id_' in col)):
                        drop_columns.append(col)

                # make the compact dataframe
                dataframe_compact = all_dataframe.drop(drop_columns, axis=1, inplace=False)

                # save the compact dataframe
                dataframe_compact.to_csv(f'{file_path}{compact_file_name}', sep =',', header=True, index=True)
       
            # also store in datastore
            if(datastore is not None):             
                   
                # also save in the registred datastore (reflected in the Azure Blob)
                results_list = []
                for path, subdirs, files in os.walk(file_path):
                    for name in files:
                        results_list.append(os.path.join(path, name))

                #upload into datastore
                datastore.upload_files(files = results_list, relative_root=file_path, 
                                        target_path= target_path, overwrite=True)   
                print(f'Upload ML modelling results to wrting results for {datastore.name}')
                
            else:
                raise Exception('Error in uploading results to datastore')
      
        except Exception as error:
            print(f'Error in {error}') 