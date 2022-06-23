# This python script is primariy used to store constants
# Python script primarily used to store constraints

#--------------------------------------------------------------------------
# 1.0 Environment variables
#--------------------------------------------------------------------------

# 1.1 Environment file
ENVIORNMENT_FILE = './environment/.env'

# 1.2 text analayics - urls
COGS_URL = 'australiaeast.api.cognitive.microsoft.com'
NLP_KEY_PHRASE_URL = '/text/analytics/v3.1/keyPhrases'
NLP_NER_URL = '/text/analytics/v3.1/entities/recognition/general'

# 1.3 python libraries
DOCKER_FILE = 'docker_dependencies.yml'
CONDA_PACKAGES = ['pandas','numpy','scikit-learn', 'scipy', 'pydub', 'nltk', 'gensim','python-dotenv']
PIP_PACKAGES = ['azureml-sdk', 'azure-cognitiveservices-speech', 'azureml-dataset-runtime[fuse,pandas]','jiwer','noisereduce', 'auto-sklearn']

#--------------------------------------------------------------------------
# 2.0 Azure Machine Learning (AML) configuration
#--------------------------------------------------------------------------

# 2.1 AML compute
COMPUTE_NAME = 'ml-kyiwasak-compute'
COMPUTE_TYPE = 'AML_COMPUTE_INSTANCE'
COMPUTE_CONFIG = 'STANDARD_D8_V3'
COMPUTE_PRIORITY = 'lowpriority'
COMPUTE_MIN_NODE = 0
COMPUTE_MAX_NODE = 4

# 2.2 Configuration for blob storages
# 2.2.1 Blob containers
RAW_CONTAINER_NAME = 'raw'
DSP_CONATINER_NAME = 'signalprocessed'
TRANSCRIBED_CONATINER_NAME = 'transcribed'
ASSESSED_CONATINER_NAME = 'assessed'

# 2.2.2 datastores & datasets
RAW_DATASTORE_NAME = 'raw_datastore'
DSP_DATASTORE_NAME = 'processed_datastore'
TRANSCRIBED_DATASTORE_NAME = 'transcribed_datastore'
ASSESSED_DATASTORE_NAME = 'assessed_datastore'
ONTOLOGY_DATASET_NAME = 'ontology'
TRUTH_TRANSCRIPTS = 'truth'
INFERENCE_DATASTORE_NAME = 'inference_datastore'
DSP_INFERENCE_DATASTORE_NAME = 'inference_processed_datastore'
TRANSCRIBED_INFERENCE_DATASTORE_NAME = 'inference_transcribed_datastore'
ASSESSED_INFERENCE_DATASTORE_NAME = 'inference_assessed_datastore'

# 2.2.3 Containers for inference
INFERENCE_CONTAINER_NAME = 'raw-inference'
DSP_INFERENCE_CONATINER_NAME = 'signalprocessed-inference'
TRANSCRIBED_INFERENCE_CONATINER_NAME = 'transcribed-inference'
ASSESSED_INFERENCE_CONATINER_NAME = 'assessed-inference'

#--------------------------------------------------------------------------
# 3.0 Directories
#--------------------------------------------------------------------------

# 3.1 general folders
RECORDINGS_FOLDER = 'recordings/'
TRUTH_TRANSCRIPTED_FOLDER = 'provided-transcripts/'
ONTOLOGY_FOLDER = 'ontology/'
INFERENCE_RECORDINGS_FOLDER = 'recordings/'
ASSESSED_FOLDER = 'assessed/'
KEY_PHRASES_FOLDER = 'key-phrases/'

# 3.2 system level - mounts & local folders
MOUNT_PATH_ROOT = 'tmp/datasets/'
MOUNT_PATH_INFERENCE_ROOT = 'tmp/datasets-inference/'
MOUNT_PATH_RECORDINGS = f'{MOUNT_PATH_ROOT}{RECORDINGS_FOLDER}'
MOUNT_PATH_INFERENCE_RECORDINGS = f'{MOUNT_PATH_INFERENCE_ROOT}{INFERENCE_RECORDINGS_FOLDER}'

MOUNT_PATH_TRUTH = f'{MOUNT_PATH_ROOT}{TRUTH_TRANSCRIPTED_FOLDER}'
MOUNT_PATH_INFERENCE_TRUTH = f'{MOUNT_PATH_INFERENCE_ROOT}{TRUTH_TRANSCRIPTED_FOLDER}'
MOUNT_PATH_KEY_PHRASES = f'{MOUNT_PATH_ROOT}{KEY_PHRASES_FOLDER}'
MOUNT_PATH_INFERENCE_KEY_PHRASES = f'{MOUNT_PATH_INFERENCE_ROOT}{KEY_PHRASES_FOLDER}'

# 3.3 results paths
RESULTS_PATH = 'results/'
RESULTS_DSP_PATH = 'DSP/'
RESULTS_TRANSCRIBE_PATH = 'transcripts/'
RESULTS_ASSESSED_PATH = 'assessed/'

# 3.4 temp directories to store results with sub-foldrs
#------------------------------------------------------------
# set the use-case
use_case = 'comms-classification'

# set the correct paths and mounting points
dsp_results_folders = f'{RESULTS_PATH}{RESULTS_DSP_PATH}{use_case}/'
transcripts_results_folder = f'{RESULTS_PATH}{RESULTS_TRANSCRIBE_PATH}{use_case}/' 
assessed_results_folder =  f'{RESULTS_PATH}{RESULTS_ASSESSED_PATH}{use_case}/' 

RECORDINGS_DATASET_NAME = f'{RAW_CONTAINER_NAME}-{use_case}'
TRUTH_DATASET_NAME = f'{TRUTH_TRANSCRIPTS}-{use_case}'
ASSESSED_DATASET_NAME = f'{ASSESSED_CONATINER_NAME}-{use_case}'

RECORDINGS_MOUNT_PATH = f'{MOUNT_PATH_ROOT}{use_case}/{RECORDINGS_FOLDER}'
TRUTH_MOUNT_PATH = f'{MOUNT_PATH_ROOT}{use_case}/{TRUTH_TRANSCRIPTED_FOLDER}'
ONTOLOGY_MOUNT_PATH = f'{MOUNT_PATH_ROOT}{use_case}/{ONTOLOGY_FOLDER}'
ASSESSED_MOUNT_PATH = f'{MOUNT_PATH_ROOT}{use_case}/{RESULTS_ASSESSED_PATH}'

# 3.5 for inferences
INFERENCE_PATH = 'results-inference/'
INFERENCE_RECORDINGS_DATASET_NAME = 'inference-recordings-dataset'
INFERENCE_TRUTH_TRANSCRIPTED_DATASET_NAME = 'inference-truth-transcripted-dataset'
INFERENCE_KEY_PHRASES_DATASET_NAME = 'inference-key-phrases-dataset'
INFERENCE_METADATA_DATASET_NAME = 'inference-metadata-dataset'
INFERENCE_ASSESSED_DATASET_NAME = 'inference-assessed-dataset'

RESULTS_RECORDINGS_INFERENCE_DSP_PATH = f'{RESULTS_DSP_PATH}{INFERENCE_RECORDINGS_FOLDER}'
RESULTS_RECORDINGS_INFERENCE_TRANSCRIBE_PATH = f'{RESULTS_TRANSCRIBE_PATH}{INFERENCE_RECORDINGS_FOLDER}'
RESULTS_RECORDINGS_INFERENCE_ASSESSED_PATH = f'{RESULTS_ASSESSED_PATH}{INFERENCE_RECORDINGS_FOLDER}'

# results paths
RESULTS_RECORDINGS_DSP_PATH = f'{RESULTS_DSP_PATH}{RECORDINGS_FOLDER}'
RESULTS_RECORDINGS_TRANSCRIBE_PATH = f'{RESULTS_TRANSCRIBE_PATH}{RECORDINGS_FOLDER}'
RESULTS_RECORDINGS_ASSESSED_PATH = f'{RESULTS_ASSESSED_PATH}{RECORDINGS_FOLDER}'

#--------------------------------------------------------------------------
# 4.0 file names
#--------------------------------------------------------------------------

# ontology filenames
GENERAL_ONTOLOGY_FILENAME = 'general-ontology.json'
RADIO_CHECK_ONTOLOGY_FILENAME = 'radio-check-ontology.json'
KEY_PHRASES_SEARCH_FILENAME = 'key-phrases-to-search.json'
MESSAGE_PROTOCOLS_FILENAME = 'message-protocols.json'
HOMOPHONE_LIST_FILENAME = 'homophone-list.txt'

METADATA_INFERENCE_FILENAME = 'master-metadata.csv'

# transcrips & NLP filenames
TRANSCRIBED_JSON_FILENAME = 'transcripted_dict.json'
TRANSCRIBED_DATAFRAME_FILENAME = 'transcripted_dataframe.csv'
TRANSCRIBED_PERFORMANCE_FILENAME = 'transcripted_performance.csv'
TRANSCRIPTS_TRUTH_FILENAME = 'transcripts-truth.csv'
CUSTOM_CORPUS_JSON_FILENAME = 'custom_corpus.json'

ASSET_NAMES_FILENAME = 'asset-names.csv'
KEY_PHRASES_FILENAME = 'key-phrases.csv'
OTHER_KEY_PHRASES_FILENAME = 'other-key-phrases.csv'
RECORDINGS_TRUTH_INFERENCE_FILENAME = 'translations.csv'

# Topic modelling fienames
CORPUS_FILENAME = 'corpus.pkl'
DICTIONARY_FILENAME = 'dictionary.gensim'
MODEL_DISPLAY_FILENAME = 'LDAModel.html'
MODEL_NAME = 'model1.gensim'
TOPIC_DATAFRAME_FILENAME = 'topic_modelling_dataframe.csv'

#--------------------------------------------------------------------------
# 5.0 Algorithm configuration
#--------------------------------------------------------------------------

# 5.1 NLP and other flags
FILE_FLAG = '.wav'
KEY_PHRASE_BATCH_SIZE = 5
NER_PHRASE_BATCH_SIZE = 5
NUM_TOPICS = 5
NUM_TOPIC_TERMS = 10


TIME_STR_LENGTH = 4
GRID_STR_LENGTH = 6
FEATURE_STR_LENGTH = 3
OBJECTIVE_STR_LENGTH = 1
VEHICLE_STR_LENGTH = 1
CODENAME_STR_LENGTH = 2
OPS_STATUS_STR_LENGTH = 1

# 5.2 Signal processing
LOW_FREQ_CUTOFF = 100 # 300
HIGH_FREQ_CUTOFF = 8000 # 5000
FILTER_ORDER = 6
THRESH_STATIONARY = 1.0
PROP_DECREASE = 1.0
FREQ_MASK_SMOOTH = 400

# 5.3 Heuristic rules
RADIO_CHECK_POS_CUTOFF = 0.66
SCENARIO_POS_CUTTOFF = 0.33

TOKEN_COUNT_CUTOFF_LOWEST = 12
TOKEN_COUNT_CUTOFF_MIDDLE = 50
TOKEN_COUNT_CUTOFF_UPPER = 100

CASEVAC_STOPS_CUTTOFF = 2
SEQUENCE_LENGTH_LOWER = 5
SEQUENCE_LENGTH_UPPER = 20

# 5.4 ML configurations
# run a logistic regression model
LOGIT_MODEL_NAME = 'logit-model'
RF_MODEL_NAME = 'randomforest-model'
GB_MODEL_NAME = 'gradientboost-model'

# ML pipelie folders, scripts and packages
EXPERIMENT_NAME = 'comms-classification-experiment'
ENVIRONMENT_NAME = 'speech2text-env'

PIPELINE_SOURCE_PATH = 'src/'
PIPELINE_SCRIPT_PATH = 'common'

FILTER_SCRIPT_FILENAME = f'{PIPELINE_SCRIPT_PATH}/filter.py'
TRANSCRIBE_SCRIPT_FILENAME = f'{PIPELINE_SCRIPT_PATH}/transcribe.py'
NLP_SCRIPT_FILENAME = f'{PIPELINE_SCRIPT_PATH}/nlp.py'
ML_SCRIPT_FILENAME = f'{PIPELINE_SCRIPT_PATH}/ml.py'

## Please specify supervised labels
MESSAGE_CLASSIFICATION_GROUP = ['OTHERS', 'UPPER', 'LOWER']

ML_KEEP_COLUMNS = [
        'word_count', 
        'num_filtered_tokens',
        'confidence',
        'num_key_phrases'
]
