import os
import json
import shutil
import http.client, urllib.request, urllib.parse, urllib.error, base64

class GeneraltUtilities():
    """
    Class mainly to perform general utilirites for the project
    """

    def __init__(self):
        super().__init__() # inherit if applicable

        self.response_data = None
        self.temp_folder = None
    
    def processRequest(self, url, cogs_service, key, body):
        """
        Function to perform url post request and get the resoense
        """
        try:
            # Request headers
            headers = {
                'Content-Type': 'application/json',
                'Ocp-Apim-Subscription-Key': key,
            }

            # Request parameters
            params = urllib.parse.urlencode({
                'model-version': 'latest',
                'showStats': 'false',
                'loggingOptOut': 'false',
            })

           
            # convert the body into json
            body = json.dumps(body)

            # format for POST request
            cogs_service_url = f'{cogs_service}?{params}'
            conn = http.client.HTTPSConnection(url)
            conn.request("POST", cogs_service_url, body , headers=headers)
            response = conn.getresponse()

            if response.status == 200:
                # decode response since in 'byte' class
                self.response_data = response.read().decode()
                conn.close()
            else:
                print(f'The response error is: {response.status}')
                raise Exception('Unsuccessful in processing the request')
                    
        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))

        finally:
            conn.close() 

    def createTmpDir(self, file_path):
        """
        Function to create directory if dioesnst exist
        """
        # checking whether folder/directory exists
        try:
            self.temp_folder = file_path
            if not os.path.exists(self.temp_folder):
                os.makedirs(self.temp_folder)

                print("folder '{}' created ".format(self.temp_folder))
        except FileExistsError:
            print("folder {} already exists".format(self.temp_folder))


    def deleteTmpDir(self, file_path):
        """
        Function to delete temp directory if dioesnst exist
        """
        try:
            self.temp_folder = file_path
            if os.path.exists(self.temp_folder):
                shutil.rmtree(shutil.rmtree(self.temp_folder, ignore_errors=False, onerror=None) )
	
                print("folder '{}' has been removed ".format(self.temp_folder))
        except OSError as e:
            print("Error: %s : %s" % (self.temp_folder, e.strerror))