# Configure the Azure ML Workspace and create datastore
#The Azure ML Workspace is key parameter for performing the text analytics, pipeline development and deployment. 
# Here, the Azure ML workspace is created (if doesnt exist) or extracts an exisitng instance
#The datastore and the asscoated datasets contain all the input data

# setup the current paths
import os, sys
import shutil, json
from dotenv import load_dotenv, find_dotenv

# azure ML core services
import azureml.core
from azureml.core import Experiment, Workspace, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication, MsiAuthentication, InteractiveLoginAuthentication
from azureml.core.dataset import Dataset
from azureml.core.datastore import Datastore
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.exceptions import  AzureMLException, ComputeTargetException, WorkspaceException

# storage related
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.storage.filedatalake import DataLakeServiceClient

# Azure ML environment
from azureml.core import Environment 
from azureml.core.environment import Environment
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies


# Azure ML pipline 
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.data import OutputFileDatasetConfig
from azureml.data.datapath import DataPath, DataPathComputeBinding

# confirming the Azure ML version
print(f'Azure ML version: {azureml.core.VERSION}')

currentDir = os.path.dirname(os.getcwd())
sys.path.append(currentDir)

# import from common setups & environments
from common.constants import *
load_dotenv(find_dotenv('./environment/.env'))

# also get storage account and key
STORAGE_ACCOUNT = os.environ.get('STORAGE_ACCOUNT')
SECRET_KEY = os.environ.get('SECRET_KEY')
MANAGED_ID = os.environ.get('MANAGED_ID')

class AzureMLConfiguration():
    """ 
    Class mainly to perform all necessary Auzre ML setups and configuration, including 
    workspace, datasets and piplines
    """

    def __init__(self, 
            workspace, 
            tenant_id, 
            subscription_id, 
            resource_group, 
            location, 
#            sp_id, 
#            sp_password
            auth
        ):
        super().__init__() # inherit if applicable

        self.workspace_name = workspace
        self.tenant_id = tenant_id
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.location = location
#        self.sp_id = sp_id
#        self.sp_password = sp_password
        self.auth = auth
        
        # workspace and other features will be assigned once configured
        self.ws = None  
        self.compute_target = None
        self.experiment = None
        self.environment = None
        self.registered_datastore = None
        self.default_datastore = None
        self.blob_list = []
        self.datsets_registered = None
        self.dataset_mount_folder = None
        self.dataset_mounted = None
        self.dataset_mounts = []

        # pipeline 
        self.aml_run_config = None
    
    def configWorkspace(self):
        """
        The Azure ML Workspace is key parameter for performing the text analytics, pipeline development and deployment. 
        Here, the Azure ML workspace is created (if doesnt exist) or extracts an exisitng instance
        The datastore and the asscoated datasets contain all the input data
        """
        # obtaining the wrkspace requires authentification. Here, the managed serive authentication is used
        # also see https://github.com/Azure/MachineLearningNotebooks/blob/master/
        # how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb
        try:
            # since it exists obtain details
            # We have 2 options for authentication: CLI-authentication and managed ID with the same codes as follows:
            self.ws = Workspace.get(name=self.workspace_name, 
                                    subscription_id=self.subscription_id, 
                                    resource_group=self.resource_group,
                                    auth = self.auth)

            # print the workspace information
            print(f'workspace name:\t{self.ws.name}')
            print(f'location:\t{self.ws.location}')
            print(f'Resource group:\t{self.ws.resource_group}')

        except:
            print('Cannot get Azure ML Workspace.')

        return self.ws


    def configCompute(self):
        """
        The Azure ML compute target is created if it doesnt existy
        """
        
        if(self.ws is not None):
            try:
                self.compute_target = ComputeTarget(workspace=self.ws, name=COMPUTE_NAME)

            except ComputeTargetException:
                config = AmlCompute.provisioning_configuration(vm_size=COMPUTE_CONFIG,
                                                            vm_priority=COMPUTE_PRIORITY, 
                                                            min_nodes=COMPUTE_MIN_NODE, 
                                                            max_nodes=COMPUTE_MAX_NODE,
                                                            identity_type=MANAGED_ID,)

                self.compute_target = ComputeTarget.create(workspace=self.ws, name=COMPUTE_NAME, provisioning_configuration=config)
                self.compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
        
            finally:
                print(f'The AML compute target name is {self.compute_target.name}')
        
        else:
            raise Exception('Error in workspace resulted to compute error')

    def configExperiment(self, experiment_name):
        """
        Function to create the ML experiments
        """
        try:
            if self.ws is not None:
                self.experiment = Experiment(workspace=self.ws, name=experiment_name)
            else:
                raise Exception('Workspace error')

        except Exception as error:
            print(f'Error in {error}')   
        
    def configEnvironment(self, environment_name):
        """
        Function to create the Azure ML environment
        """       
        try:
            if self.ws is not None:
                self.environment = Environment(name = environment_name)
                
                # create the conda dependencies wich includes both native pip and conda
                pip_packages=['azureml-sdk', 'azure-cognitiveservices-speech', 'azureml-dataset-runtime[fuse,pandas]', 'jiwer','noisereduce', 'azure-storage-blob', 'azure-storage-file-datalake']
                conda_packages=['pandas','scikit-learn', 'scipy', 'pydub', 'nltk', 'gensim','python-dotenv','numpy']
                
                # note:  If pin_sdk_version is set to true, pip dependencies of the packages distributed as a part of 
                # Azure Machine Learning Python SDK will be pinned to the SDK version installed in the current environment.
                conda_dependencies = CondaDependencies.create(pip_packages=pip_packages, conda_packages=conda_packages,pin_sdk_version=False)
               
                # set to the environment
                self.environment.python.conda_dependencies = conda_dependencies
                    
                # Register environment to re-use later
                self.environment.register(workspace=self.ws)

            else:
                raise Exception('Workspace error')

        except Exception as error:
            print(f'Error in {error}')
         
    def configDataStore(self,datastore, container_name):
        """
        Function to configure the Azure ML datastore for registering (though Azure ML workspace instance)
        to use the input voice data and other files as datasets
        """
        if(self.ws is not None):       

            ## If container doesn't exist in blob storage, generate it.
            conn_str = f'DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT};AccountKey={SECRET_KEY};EndpointSuffix=core.windows.net'
            blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            container_list = [c['name'] for c in blob_service_client.list_containers()]
            if container_name in container_list:
                print("Container {} already exists.".format(container_name))
            else:
                blob_service_client.create_container(container_name)
                print('Created {} container in {}'.format(container_name, STORAGE_ACCOUNT))                

            try:
                # Register datastore to Azure ML
                self.registered_datastore = Datastore.register_azure_blob_container(workspace=self.ws, 
                                    datastore_name=datastore, 
                                    container_name=container_name, 
                                    account_name=STORAGE_ACCOUNT, 
                                    account_key=SECRET_KEY,
                                    skip_validation = True,
                                    overwrite=True,
                                    subscription_id=self.subscription_id,
                                    resource_group=self.resource_group
                                    )

                # workspace default datastore                       
                self.default_datastore = self.ws.get_default_datastore()

            except Exception as error:
                print(f'Error in {error}')
                
            finally:
                print(f'The Azure ML datastore details are {self.registered_datastore}')
                return self.registered_datastore
        else:
            raise Exception('Error in workspace resulted to datastore error')
            return None

    def make_directory_in_container(self, container_name, directory):
        """
        Make directory in container of blob storage
        """
        conn_str = f'DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT};AccountKey={SECRET_KEY};EndpointSuffix=core.windows.net'
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        container_list = [c['name'] for c in blob_service_client.list_containers()]
        ## Check existence of container
        if container_name in container_list:
            datalake_service_client = DataLakeServiceClient.from_connection_string(conn_str)

            try:
                ## Make specified directory
                file_system_client = datalake_service_client.get_file_system_client(container_name)
                directory_client = file_system_client.create_directory(directory)
            except Exception as error:
                    print(f'Error in {error}')
        else:
            print(f'Container {container_name} does not exist.')


    def getBlobFiles(self, container_name, folder_name=None):
        """
        Function to get the list and names of the positive comms audio files form the blob store
        """
        try:
            blob_list_refined = []
            # Instantiate a BlobServiceClient using a connection string
            conn_str = f'DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT};AccountKey={SECRET_KEY};EndpointSuffix=core.windows.net'
            blob_service_client = BlobServiceClient.from_connection_string(conn_str)

            # Instantiate a ContainerClient
            container_client = blob_service_client.get_container_client(container=container_name)

            # list the blobs within this container
            # perform filter based on folder_name (if required)
            self.blob_list = list(container_client.list_blobs())
            
            if(folder_name is not None):
                for blob in self.blob_list:
                    if(blob.name.startswith(folder_name) and (blob.name.endswith('.wav') or blob.name.endswith('.txt'))):
                        blob_list_refined.append(blob) 

        except Exception as error:
            print(f'Error in {error}')
            blob_list_refined = []
            
        finally:
            blob_service_client.close()
            return blob_list_refined

    def getDatastoreFile(self, datastore_name, prefix_path, file_name, target_path):
        """
        Function to get the datastore file as dictionary
        """
        try:
            # initialise
            dictionary = dict()

            # get the datastore object
            datastore = Datastore.get(self.ws, datastore_name=datastore_name)  

            # download to temp results folder
            datastore.download(target_path=target_path, prefix=f'{prefix_path}{file_name}', overwrite=True)

            # read as python dictionary
            target_file_name = f'{target_path}{prefix_path}{file_name}'
            with open(target_file_name, "r") as read_file:
                dictionary = json.load(read_file)

        except Exception as error:
            print(f'Error in {error}')

        finally:
            return dictionary


    def configDatasets(self, datastore, file_path, dataset_name, description):
        """
        Function to register the Azure ML Datasets (based on Datastore) for the
        trabscribe UHF recordings and other files
        """
        try:
            # TODO: generate subdirectory `file_path` in container of blob storage, if needed

            # the battle voice dataset register - based on number of files
            datsets_registered = None

            # get all the audio files in datastore
            datastore_files =  Dataset.File.from_files(path = (datastore, file_path), is_file=False) 

            #register the dataset
            datastore_files.register(workspace=self.ws, name= dataset_name, description=description, create_new_version=True)

            # store these registred datasets
            datsets_registered = Dataset.get_by_name(self.ws, name=dataset_name)

            print('Registered datasets are:', datsets_registered)
            
        except Exception as error:
            print(f'Error in {error}')
                
        finally:
            return datsets_registered
    
    
    def mountDatasetsBatched(self, datasets, mount_path='tmp/datasets/'):
        """
        Function to read the registered datasets in Azure ML pipeline and mount on to local
        field systme as tmp
        """
               
        if(self.ws is not None):
            try:
                dataset_mounted = []
                # loop through the datasets and mount on to tmp folder
                for dataset_name in datasets:
                    print(f'Mounting: {dataset_name}')
                    dataset = Dataset.get_by_name(workspace=self.ws, name=dataset_name, version='latest')
                    
                    #prepare for simplified mount point names
                    dataset_mount_name = dataset_name[dataset_name.find('/')+1:dataset_name.find('.')]
                    dataset_mount_path = f'{mount_path}{dataset_mount_name}'
                    
                    dataset_mount = dataset.mount(mount_point=dataset_mount_path)
                    self.dataset_mounts.append(dataset_mount)

                # mount the datasets
                if(len(self.dataset_mounts) != 0):
                    for dataset_mount in self.dataset_mounts:
                        if not os.path.ismount(dataset_mount.mount_point):
                            print(f'Starting mounting point: {dataset_mount.mount_point}')
                            dataset_mount.start()
            
                # store the mounted files point
                dataset_mounted = os.listdir(mount_path)
                
                print(f'Dataset mounitng completed on: {mount_path}')
                print(f'Mounted files are: {dataset_mounted}')

            except Exception as error:
                print(f'Error in {error}')
            
            finally:
                return dataset_mounted
        else:
            raise Exception('Error in workspace resulted to datastore error')
            return None

    def mountDatasets(self, datasets_registered, mount_path='tmp/datasets/'):
        """
        Function to read the registered datasets in Azure ML pipeline and mount on to local
        field systme as tmp
        """
               
        if(self.ws is not None):
            ## If mount_path is already moounted, unmount
            if os.path.ismount(mount_path):
                return None, os.listdir(mount_path)
            else:
                try:
                    # prepare the mount pint
                    dataset_mounts_context = datasets_registered.mount(mount_point=mount_path)
                
                    # mount the datasets
                    print(f'Starting mounting: {dataset_mounts_context.mount_point}')
                    dataset_mounts_context.start()
                    print(f'Dataset mounting completed on: {mount_path}')

                    return dataset_mounts_context, os.listdir(mount_path)

                except Exception as error:
                    print(f'Error in {error}')
            
        else:
            raise Exception('Error in workspace resulted to datastore error')
            return None


    def unmountDatasets(self, datasets_mount_point, mount_path, remove_path=False):
        """
        Function to unmount the datasets from local file
        """  
        try:
            
            # check to see if mounted
            if os.path.ismount(mount_path):
                print(f'Unmounting the datasets')
                datasets_mount_point.stop()
            else:
                print('No dataset files to unmount')

            if(remove_path):
                # also remove the directory (if tmp exists)
                if os.path.exists(mount_path):
                    shutil.rmtree(mount_path)

        except Exception as error:
            print(f'Error in {error}')

    def unmountDatasets(self, mount_path, remove_path=False):
        """
        Function to unmount the datasets from local file
        """  
        try:
            
            # check to see if mounted
            if os.path.ismount(mount_path):
                print(f'Unmounting the datasets')
                cmd = f'sudo umount {mount_path}'
                os.system(cmd)
            else:
                print('No dataset files to unmount')

            if(remove_path):
                # also remove the directory (if tmp exists)
                if os.path.exists(mount_path):
                    shutil.rmtree(mount_path)

        except Exception as error:
            print(f'Error in {error}')

    def configPipeline(self, docker_path, conda_packages, pip_packages):
        """
        Function to configure the Azure ML pipeline and its parameters
        """
        try:
            # may require to set the azure ml snapshot size
            azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 1000000000

            # setup the pipeline run configuration
            self.aml_run_config = RunConfiguration()                                    
            self.aml_run_config.target = self.compute_target

            # setup docker configuration
            self.aml_run_config.environment.docker.base_image = None                      
            self.aml_run_config.environment.docker.base_dockerfile = docker_path           

            # Add some packages required by the pipeline scripts step to the environment configuration file
            self.aml_run_config.environment.python.user_managed_dependencies = False     
            self.aml_run_config.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=conda_packages, 
                                                                        pip_packages=pip_packages, pin_sdk_version=False) 

        except Exception as error:
            print(f'Error in {error}')

    def configPipelineInputData(self, datastore, path_on_datastore):
        """
        Function to configure the pipeline input data path
        """
        try:
            datapath_pipeline_param = None
            datapath_input = None

            if datastore is not None:

                # setup the data path
                data_path = DataPath(datastore=datastore, path_on_datastore=path_on_datastore)
                
                # setup the parapeter and the input path
                datapath_pipeline_param = PipelineParameter(name='Datastore where raw audio files are located', default_value=data_path)
                datapath_input = (datapath_pipeline_param, DataPathComputeBinding(mode='mount'))

            else:
                raise Exception("Datastore is none'")
        except Exception as error:
            print(f'Error in {error}')
        
        finally:
            return datapath_pipeline_param, datapath_input


    def configPipelineOutputData(self, name, destination):
        """
        Function to configure the pipeline output datasets
        """
        try:
            output_ds = None

            # configure the output dataset
            output_ds = OutputFileDatasetConfig(name=name, destination=destination).as_mount() 
    
        except Exception as error:
            print(f'Error in {error}')

        finally:
            return output_ds
    
    def configPipelineParameter(self, description, default_value):
        """
        Function to configure and resturn the pipelien paramaters
        """
        try:
            #setup the pipeline parameter, with default value
            pipeline_parameter = None
            pipeline_parameter = PipelineParameter(name=description, default_value=default_value)

        except Exception as error:
            print(f'Error in {error}')

        finally:
            return pipeline_parameter

    def buildPipeline(self, workspace, script_steps):
        """
        Function to buld the pipeline based on thhe provided steps
        """
        try:
            pipeline = None

            if workspace is not None:
                # build the pipeline
                pipeline = Pipeline(workspace=workspace, steps=script_steps)

            else:
                raise Exception('Azure ml workspace is nonw')

        except Exception as error:
            print(f'Error in {error}')

        finally:
            return pipeline

    def submitPipeline(self, experiment, pipeline, regenerate_outputs=True):
        """
        Function to submit and run the Azure ML pipeline
        """
        try:
            run = None

            if experiment is not None:
                # submit and run the experiment
                run = experiment.submit(pipeline=pipeline, regenerate_outputs=regenerate_outputs)

            else:
                raise Exception('Experiment is not configured')
    
        except Exception as error:
            print(f'Error in {error}')

        finally:
            return run
