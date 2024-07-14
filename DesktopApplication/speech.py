
import pyttsx3
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
from code_summarization import ASTProcessor

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
        self.active = False

        self.interactive = False
        self.voice_engine = None

        self.summarizer = None

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
            print(f"Error initializing Vosk: {e}")
            self.voice_recognition_tool = 'google'
            self.initialize_google_sr()

    def process_command(self, command):
        print(f"Command: {command}")
        # process the command
        try:
            intent, response = self.command_intent.process_command(command)
            # print(self.command_intent.process_command(command))
            print(f"Intent: {intent}")
            print(f"Response: {response}")
            # print(f"Response: {type(response)}")
            intent = 'summary'
            if intent == 'summary':
                response['message'] = self.get_file_summary()

            if self.interactive:
                self.interactive_response(response)

        except Exception as e:
            print(f"Error in processing command: {e}")

    def get_file_summary(self):
        self.summarizer = ASTProcessor({})

        s = self.summarizer.get_summary()
        print(s)
        return s

    def activate_interactive(self,):
        print('activated interactive')
        self.interactive = True
        pyttsx3.speak("Interactive mode activated")

    def deactivate_interactive(self):

        pyttsx3.speak("Interactive mode deactivated")

        self.interactive = False

    def interactive_response(self, response):
        if 'message' in response:
            pyttsx3.speak(response['message'])
        else:
            print(response)

    def recognize(self):
        # based on the speech recognition method used
        if self.voice_recognition_tool == 'google':
            while self.active:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source,
                                                             duration=1)
                    audio = self.recognizer.listen(source)
                    try:
                        r = self.recognizer.recognize_google(audio)

                        print(r)
                        if (r != 'hey robin'):
                            # process the command
                            try:
                                # self.command_intent.process_command(r)
                                self.process_command(r)
                            except Exception as e:
                                print(f"Error in processing command: {e}")
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
                    result = json.loads(result)
                    # print(result)
                    if 'text' in result and result['text'].strip() != "" and result['text'].strip() != 'hey Robin':
                        # process the command
                        print(result['text'])
                        try:
                            self.process_command(r)

                            # self.command_intent.process_command(result['text'])
                        except Exception as e:
                            print(f"Error in processing command: {e}")
