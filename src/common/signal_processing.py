import os, sys
import numpy as np
currentDir = os.path.dirname(os.getcwd())
sys.path.append(currentDir)
sys.path.append('../')
sys.path.append('./')
sys.path.append('./../')

from common.constants import *

# audio file processing
from scipy.io import wavfile
from scipy import signal
import noisereduce as nr
from pydub import AudioSegment
import matplotlib.pyplot as plt

from azureml.core import Dataset

class SignalProcessing():
    """
    This class focuses on performing various signal processing filters and 
    algorithms on the raw datasets to filter out noises such as: bandpass filtering 
    white noise cancellations and notch filtering
    """

    def __init__(self):
        super().__init__() # inherit if applicable

        self.audio_file_name = None
        self.audio_file_path = None
        self.audio_data = None
        self.filtered_audio_data = None
        self.noise_reduced_audio_data = None
        self.low_freq_cut = None
        self.high_freq_cut = None
        self.sampling_rate = None
        self.order = None
        

    def configFilter(self, low_freq_cut, high_freq_cut, order):
        self.low_freq_cut = low_freq_cut
        self.high_freq_cut = high_freq_cut
        self.order = order
    
    def __stereoToMono(self, audiodata):
        """
        Function (private) to convert Stereo to single mono
        """
        try:
            newaudiodata = []
            audio_data_mono = None

            for i in range(len(audiodata)):
                # perform the averaging
                d = (audiodata[i][0]/2 + audiodata[i][1]/2)
                newaudiodata.append(d)

            # convert to numpy array
            audio_data_mono = np.array(newaudiodata, dtype='int16')

        except Exception as error:
            print(f'Error in {error}')

        finally:
            return audio_data_mono

    def readAudioFile(self, audio_file_name, mount_path, stereo=True):
        """
        Function to read the audio file as digitized value
        """
        try:
            # set the name and path
            self.audio_file_name = audio_file_name
            self.audio_file_path = f'{mount_path}{self.audio_file_name}'
            
            # read the audio file as byte data and the sampling rate
            if self.audio_file_name is not None:
                self.sampling_rate, self.audio_data = wavfile.read(self.audio_file_path)

                # if steror then must average the channels (call private function)
                if stereo:
                    self.audio_data = self.__stereoToMono(self.audio_data)    

            else:
                raise Exception(f'Error occurred reading waqve file {self.audio_file_path}')

        except Exception as error:
            print(f'Error in {error}')    


    def processButterworthFilter(self, data):
        """
        Function to setup the butterworth bandpass filter
        """
        try:
            y = None
            nyq_frequency = 0.5 * self.sampling_rate
            low = self.low_freq_cut / nyq_frequency
            high = min(self.high_freq_cut / nyq_frequency, 0.98)

            #call the butterworh filter
            b, a = signal.butter(self.order, [low, high], btype='band', analog=False)    
            y = signal.filtfilt(b, a, data)
            
        except Exception as error:
            print(f'Error in {error}')     

        finally:
            return y

    def butterworthFilter(self):
        """
        Function to filter the audio file using butterworth filter
        """
        try:

            self.audio_data = np.apply_along_axis(self.processButterworthFilter, 0, self.audio_data).astype('int16')

        except Exception as error:
            print(f'Error in {error}')


    def fftFilter(self, stationary, thresh_stationary, prop_decrease, freq_smooth):
        """
        Function to perform fast-fourier-transformations (fft) on the filtered
        audio wav file
        """
        try:
            ## audio data has plural channel like stereo?
            if len(self.audio_data.shape) > 1:
                self.audio_data = self.audio_data[:, 0] ## Pick up the first sound sequence
            # apply stationary fft noise reductions
            self.audio_data = nr.reduce_noise(y = self.audio_data, sr=self.sampling_rate,
                                                n_std_thresh_stationary=thresh_stationary,
                                                prop_decrease=prop_decrease,
                                                freq_mask_smooth_hz=freq_smooth,
                                                stationary=stationary)

        except Exception as error:
            print(f'Error in {error}')

         
    def saveAudioFiltered(self, filtered_audio_file_path, volume=0, fix_Name=True):
        """
        Function to save the DSP (filtered audio files) including
        the volume (if required)
        """
        try:
            # save the audio data with the sampling rates
            filtered_audio_file_name = self.audio_file_name[:self.audio_file_name.find('.wav')]

            # check to see if further string processing must be performed
            # in this case remove any training whitespaces
            if fix_Name:
                filtered_audio_file_name = filtered_audio_file_name.rstrip()

            save_path = f'{filtered_audio_file_path}{filtered_audio_file_name}_filtered.wav'
            wavfile.write(save_path, self.sampling_rate, self.audio_data)

            # check if the adudio file needs to be amplified
            if (volume > 0):
                audio_data_noise_reduced = AudioSegment.from_wav(save_path)

                # add the volumne in dB
                audio_data_noise_reduced = audio_data_noise_reduced + volume
                # save the file back into temp folder
                audio_data_noise_reduced.export(save_path, format='wav')
   
        except Exception as error:
            print(f'Error in {error}')
            

    def saveAudioFilteredtoDatastore(self, datastore, filtered_audio_file_path, target_path):
        """
        Function to save the filtered audio files to datastore. This is will streamlined as
        a generic call in future
        """
        try:
            if(datastore is not None):
                ## Setup target datastore and its path
                target = (datastore, target_path)
                ## Upload each file into the target datastore
                Dataset.File.upload_directory(src_dir=filtered_audio_file_path,
                            pattern='*_filtered.wav',
                            target=target)

                print(f'Upload results to wrting results for {datastore.name}')
            
            else:
                raise Exception('Error in uploading results to datastore')

        except Exception as error:
            print(f'Error in {error}')


    def filterPlot(self):
        """
        Function ot plot the filter with gain response
        """
        try:
            # plot the filter
            nyq_frequency = 0.5 * self.sampling_rate
            low = self.low_freq_cut / nyq_frequency
            high = self.high_freq_cut / nyq_frequency

           
            b, a = signal.butter(self.order, [low, high], btype='band', analog=False) 
            w, h = signal.freqz(b, a, worN=2000)
            plt.plot((self.sampling_rate * 0.5 / np.pi) * w, abs(h), label="order = %d" % self.order)

            plt.plot([0, 0.5 * self.sampling_rate], [np.sqrt(0.5), np.sqrt(0.5)],
                        '--', label='sqrt(0.5)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Gain')
            plt.grid(True)
            plt.legend(loc='best')   
            
        except Exception as error:
            print(f'Error in {error}')

    def audioPlot(self, filtered_audio_file_path):
        """
        Function to plot the audio files in time domain to 
        analyse the gain response
        """
        try:

            fig, ax = plt.subplots(figsize=(20,3))
            
        except Exception as error:
            print(f'Error in {error}')