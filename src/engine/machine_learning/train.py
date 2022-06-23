import pandas as pd
import numpy as np
import argparse
import os, sys
import glob
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from azureml.core import Run
from azureml.core.model import Model

# append the correct paths
sys.path.append(os.path.dirname(__file__) + './../')

from common.constants import MESSAGE_CLASSIFICATION_GROUP, ML_KEEP_COLUMNS
from common.azureml_configuration import AzureMLConfiguration

def prepareMLDataframe(dataframe):
        """
        Function to further reduce the ML dataframe for modelling
        """
        try:
            ml_datraframe_independents = pd.DataFrame()
            ml_datraframe_dependents = pd.DataFrame()

            if 'ground_truth_class' in dataframe.columns:
                dataframe['ground_truth_class_num'] = dataframe['ground_truth_class'].apply(lambda x: 0 if x == MESSAGE_CLASSIFICATION_GROUP[0] else 
                                            (1 if x == MESSAGE_CLASSIFICATION_GROUP[1] else
                                            (2 if x == MESSAGE_CLASSIFICATION_GROUP[2] else  0))).astype(np.int8)

            # Columns to keep
            keep_column_list = ML_KEEP_COLUMNS                  ## for independent variables
            keep_column_list.append('ground_truth_class_num')   ## for dependent variable

            # create an ML dataframe
            if dataframe is not None:
                ml_dataframe_reduced = dataframe[keep_column_list]

                # assign the dependant & independant variables
                ml_datraframe_independents = ml_dataframe_reduced.drop(columns='ground_truth_class_num', axis=1)
                ml_datraframe_dependents = ml_dataframe_reduced['ground_truth_class_num']

            else:
                raise Exception('dataframe does not exist')    

        except Exception as error:
            print(f'Error in {error}') 

        finally:
            return ml_datraframe_independents, ml_datraframe_dependents


def prepare_train_test_split(df_independent, df_dependent, test_size=0.33):
        """
        Function to split the ml dataframe into trainng and testing
        """
        try:
            X_train = pd.DataFrame()
            X_test = pd.DataFrame()
            y_train = pd.DataFrame()
            y_test = pd.DataFrame()
            train = pd.DataFrame()
            test = pd.DataFrame()

            if (df_independent is not None and df_dependent is not None):
                X = df_independent
                y = df_dependent

                X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=32)

                # also process the train and test
                # train set all 
                train = pd.concat([X_train, y_train], axis=1, ignore_index=False)
                # test set all
                test = pd.concat([X_test, y_test], axis=1, ignore_index=False)

            else:
                raise Exception ('Dataframe is empty')

        except Exception as error:
            print(f'Error in {error}') 

        finally:
            return X_train, X_test, y_train, y_test, train, test

# obtain the current run context
run = Run.get_context()

# extract the arguments sent to this file script
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--file-name', type=str, dest='file_name', help='file name to extract')
parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
parser.add_argument('--workspace', type=str, dest='workspace', default='workspace', help='AML Workspace')
parser.add_argument('--tenant_id', type=str, dest='tenant_id', default='tenant_id01', help='tenant ID')
parser.add_argument('--subscription_id', type=str, dest='subscription_id', default='Subscription', help='Subscription ID')
parser.add_argument('--resource_group', type=str, dest='resource_group', default='resource_group', help='Resource group')
parser.add_argument('--location', type=str, dest='location', default='us', help='location')
parser.add_argument('--sp_id', type=str, dest='sp_id', default='id', help='Service Principal ID')
parser.add_argument('--sp_password', type=str, dest='sp_password', default='secret', help='Secret of Service Principal')
parser.add_argument('--datastore', type=str, dest='datastore', default='datastore', help='datastore associated to Azure ML')
parser.add_argument('--container_name', type=str, dest='container', default='container', help='Container in datastore')
args = parser.parse_args()

# extract arguments
data_folder = args.data_folder
file_name = args.file_name
reg = args.reg
workspace = args.workspace
tenant_id = args.tenant_id
subscription_id = args.subscription_id
resource_group = args.resource_group
location = args.location
sp_id = args.sp_id
sp_password = args.sp_password
datastore_name = args.datastore
container_name = args.container


# set the correct path for the ml dataframe
ml_dataframe_file = f'{data_folder}/{file_name}'

# read the dataframe
#-------------------
print(f'{os.getcwd()} is current directory.')
ml_dataframe_all = pd.read_csv(ml_dataframe_file, encoding = 'unicode_escape', engine ='python')

# Configure Azure ML workspace
#-------------------
azuremlConfig = AzureMLConfiguration(workspace=workspace
                                    ,tenant_id=tenant_id
                                    ,subscription_id=subscription_id
                                    ,resource_group=resource_group
                                    ,location=location
                                    ,sp_id=sp_id
                                    ,sp_password=sp_password)

## Azure ML Workspace
ws = azuremlConfig.configWorkspace()
## datastore for artifacts
result_datastore = azuremlConfig.configDataStore(datastore=datastore_name, container_name=container_name)

# prepare the datafarme
#---------------------
ml_datraframe_independents, ml_datraframe_dependents = prepareMLDataframe(ml_dataframe_all)

# perform the train / test split
#-------------------------------
X_train, X_test, y_train, y_test, train, test = prepare_train_test_split(ml_datraframe_independents, ml_datraframe_dependents, 0.33)


# Peform Modelling
#-----------------
print(f'Model 1 - Logistic Reghression with regularization rate of: {reg}')

# initialiose the model
model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0/reg, solver="liblinear", max_iter=100, multi_class="auto", random_state=42)

#model = GradientBoostingClassifier(criterion= 'friedman_mse',learning_rate=0.1, n_estimators=100,  
                                        #max_depth=3, min_impurity_decrease=0.0, validation_fraction=0.1, 
                                        #n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
# fit the logit to the training set
model.fit(X=X_train, y=y_train)

# perform predictions
print('Predict the test set')
y_predict = model.predict(X=X_test)

# perform evaluation
#-------------------
cm = confusion_matrix(y_true=y_test, y_pred=y_predict)

TN, FP, FN, TP = cm.ravel()

accuracy = accuracy_score(y_true=y_test, y_pred=y_predict)
precision = precision_score(y_true=y_test, y_pred=y_predict, average='binary')
recall = recall_score(y_true=y_test, y_pred=y_predict, average='binary')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# log various metric
#-------------------
run.log(name='regularization rate', value=np.float(reg), description='regularization rate use for logistic reg. model')
run.log(name='accuracy', value=accuracy, description='accuracy of logistic reg model')
run.log(name='precision', value=precision, description='precision of logistic reg model')
run.log(name='recall', value=recall, description='recall of logistic reg model')

# prepare the dataframe to write results
#---------------------------------------
test['interaction_predicted'] = y_predict

# store the results
test.to_csv('ML_dataframe_predicted.csv', sep =',', header=True, index=True)
result_datastore.upload_files(files =['ML_dataframe_predicted.csv'], 
                            relative_root='/', 
                            target_path= '/ML_prediction_result/', 
                            overwrite=True)

# note file saved in the outputs folder is automatically uploaded into experiment record
model_name = 'best-model.pkl'

joblib.dump(value=model, filename=model_name)
result_datastore.upload_files(files =[model_name], 
                            relative_root='/', 
                            target_path= '/ML_models/', 
                            overwrite=True)

Model.register(model_path=model_name,
                model_name='best_model',
                workspace=ws)

print('Training completed')

