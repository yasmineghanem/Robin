
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import pyaudio
import requests
import importlib.util
from api import *
import os
import sys
import json
import tensorflow as tf
import torch
import numpy as np

np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(1)

# Add the DesktopApplication directory to the Python path
desktop_application_path = os.path.abspath('.')
sys.path.append(desktop_application_path)

# Add the CommandIntent/final_files directory to the Python path
command_intent_path = os.path.abspath('../CommandIntent/final_files')
sys.path.append(command_intent_path)

# Path to the command_intent module
module_name = "command_intent"
module_file_path = os.path.join(command_intent_path, "command_intent.py")
spec = importlib.util.spec_from_file_location(module_name, module_file_path)
command_intent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(command_intent)

# Import the CommandIntent class
CommandIntent = command_intent.CommandIntent

# Correctly resolve paths relative to the script's location
intent_model_path = os.path.abspath(
    '../CommandIntent/models/intent_detection_model_2.h5')
ner_model_path = os.path.abspath(
    '../CommandIntent/models/constrained_ner_model_2.pth')


class SpeechRecognition:
    def __init__(self, gui):
        self.gui = gui
        self.command_intent = CommandIntent(intent_model_path, ner_model_path)
        self.api = APIController()

        # read config file
        with open('./config.json') as f:
            self.__config = json.load(f)
            self.voice_recognition_tool = self.__config['voice_recognition']
            # check voice recognition
            if self.__config['voice_recognition'] == 'google':
                self.initialize_google_sr()

            elif self.__config['voice_recognition'] == 'vosk':
                self.initialize_vosk_sr()

    def initialize_google_sr(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(
            device_index=1, sample_rate=48000, chunk_size=2048
        )
        # with self.microphone as source:
        #     self.recognizer.adjust_for_ambient_noise(source)
        self.active = True
        print('initialized google')

    def initialize_vosk_sr(self):
        try:
            self.model = Model("./Assets/voice_recognition_models/" +
                               ("v_2" if self.__config['light_weight'] == False else "v_1"))
            self.recognizer = KaldiRecognizer(self.model, 16000)
            self.mic = pyaudio.PyAudio()
            self.stream = self.mic.open(
                format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
            self.active = True

            print('initialized vosk')

        except Exception as e:
            self.initialize_google_sr()

    def recognize(self):
        print("recogninzing")
        # based on the speech recognition method used
        if self.voice_recognition_tool == 'google':
            while self.active:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(source)
                    try:
                        r = self.recognizer.recognize_google(audio)
                        print(r)

                        # process the command
                        self.command_intent.process_command(r)
                    except sr.UnknownValueError:
                        print("Google Speech Recognition could not understand audio")
                    except sr.RequestError as e:
                        print(
                            "Could not request results from Google Speech Recognition service; {0}".format(e))
        elif self.voice_recognition_tool == 'vosk':
            while self.active:
                data = self.stream.read(8192)
                if len(data) == 0:
                    break
                if self.recognizer.AcceptWaveform(data):
                    print("RECOGNIZED")
                    result = self.recognizer.Result()
                    # if 'text' in result:
                    print(result)
                    # return result['text']
                    # else:
                    #     return None
