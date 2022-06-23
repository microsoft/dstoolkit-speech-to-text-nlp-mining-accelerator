# This Python script i sused to specify the various communication protocols
# used in C3PO including 'radio check' and 'CASEVAC' 
# setup the current paths
import os, sys
currentDir = os.path.dirname(os.getcwd())
sys.path.append(currentDir)

# system related
import json
from itertools import product


class Ontology():
    """
    This class focuses on the creating the ontology and the corpus for battle command
    specific dictionary and key phrases
    """
    
    def __init__(self):
        super().__init__() # inherit if applicable
        
        self.main_dictionary = dict()
        self.key_phrase_search_dictionary = dict()
        self.homophone_list = list(tuple())
        
        # this will be used to translate word numbers into numbers
        self.word_to_num_dict = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'zero' : '0'}


    def configOntology(self, file_path, ontology_list, ontology_to_enhance=None):
        """
        Function to read the dictionaries and create the main dictionary
        """
        try:
            
            if len(ontology_list) > 0:
                # preocess the path and list
                ontology_list = [file_path + str(x) for x in ontology_list]

                # loop thruogh the fil paths and updat edictionary
                for file_path in ontology_list:
                    # read the ontology as python dictionary
                    with open(file_path, "r") as read_file:
                        temp_dictionary = json.load(read_file)

                        # check to see if enhancements are required
                        if ontology_to_enhance is not None:
                            if ontology_to_enhance in file_path:
                                temp_dictionary = self.__enhanceOntology(temp_dictionary)     
        
                    # append to main dictionary
                    self.main_dictionary.update(temp_dictionary)

            else:
                raise Exception('FIle paths not provided')

        except Exception as error:
            print(f'Error in {error}')


    def __enhanceOntology(self, base_dictionary): 
        """
        Function to create an enhanced radio-cehck dictionary based on 
        the base dictionary
        """
        try:
            # initialise a dictionary to return
            enhanced_dictionary = dict()

            # 1. call sign combination - in this case l-n & n-l
            call_signs = list()
            call_sign_letters = base_dictionary.get('call-signs-letters')
            call_sign_numbers = base_dictionary.get('call-signs-numbers')
            call_sign_letters_map = map(( lambda x: x + ' '), call_sign_letters)
            call_sign_numbers_map = map(( lambda x: x + ' '), call_sign_numbers)

            # letter - number combination
            for call_sign_pair in product(call_sign_letters_map, call_sign_numbers):
                call_signs.append(''.join(call_sign_pair))

            # add to the dictionary
            enhanced_dictionary['call-signs'] = call_signs

            # number - letter comniation
            call_signs = list()

            for call_sign_pair in product(call_sign_numbers_map, call_sign_letters):
                call_signs.append(''.join(call_sign_pair))

            # update the call sings elements
            enhanced_dictionary['call-signs'].extend(call_signs)

            # 2. add signal strength
            signals = list()
            signal_strength = base_dictionary.get('signal-strength')
            signal_readability = base_dictionary.get('readability')
            signal_strength = map(( lambda x: x + ' and '), signal_strength)
            for signal_pair in product(signal_strength, signal_readability):
                signals.append(''.join(signal_pair))   

            # add to the dictionary
            enhanced_dictionary['signal-readability'] = signals
            
            # update other fileds
            enhanced_dictionary.update(base_dictionary)
            
        except Exception as error:
            print(f'Error in {error}')
        finally:
            return enhanced_dictionary

    def configKeyPhraseSearch(self, file_path):
        """
        Function to configure the key phrases required to search and surface 
        as insights
        """
        try:
            # read the json and convert to python dictionary
            if len(file_path) > 0: 
                with open(file_path, "r") as read_file:
                    self.key_phrase_search_dictionary = json.load(read_file)
            else:
                raise Exception ('Key phrase to search file path does not exist')
        
        except Exception as error:
            print(f'Error in {error}')


    def configHomophone(self, file_path):
        """
        Function to configure the homophone list. I.e. list of words
        to replace since it sound the same
        """
        try:
            # read the list, and remove whitespaces around
            if len(file_path) > 0: 
                with open(file_path, 'r') as f:
                    content = f.read().strip() 
                    self.homophone_list = eval(content)     
            else:
                raise Exception ('Homophone file path does not exist')
        
        except Exception as error:
            print(f'Error in {error}')


