
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import pyaudio
import requests
import importlib.util
from api import *
import os
import sys
# from CommandIntent.final_files.command_intent import CommandIntent

# Add the directory containing the module to the Python path
# sys.path.append(os.path.abspath('../CommandIntent/final_files'))

# # Now you can import your module
# import command_intent

# # Use the module's functionality
# module.some_function()


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
intent_model_path = os.path.abspath('../CommandIntent/models/intent_detection_model.h5')
ner_model_path = os.path.abspath('../CommandIntent/models/full_ner_model.pth')

class SpeechRecognition:
    def __init__(self, gui):
        # self.recognizer = sr.Recognizer()
        # self.microphone = sr.Microphone()
        self.gui = gui
        self.command_intent = CommandIntent(intent_model_path, ner_model_path)

        # vosk
        self.model = Model("./Assets/voice_recognition_models/v_2")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.mic = pyaudio.PyAudio()

        self.stream = self.mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        self.active = True

        self.api = APIController()

    def activate(self):
        self.active = True
        self.recognize()

    def deactivate(self):
        self.active = False

    def recognize(self):
        # vosk
        while self.active:
            data = self.stream.read(8192)
            if len(data) == 0:
                break
            if self.recognizer.AcceptWaveform(data):
                r = self.recognizer.Result()

                # print(r)

                response = self.command_intent.process_command(r['text'])
                # response = self.command_intent.process_command('cast the variable x to integer')

                

                # # declare variable
                # self.api.declare_variable('new_variable', 5)

                # # declare function
                # self.api.declare_function('new_function',  [
                #     {
                #         "name": "x_variable",
                #         "value": "test"
                #     },
                #     {
                #         "name": "y"
                #     }
                # ])

                # # if condition
                # self.api.conditional([{"keyword": 'if', "condition": [
                #     {
                #         "left": "x",
                #         "operator": ">",
                #         "right": "5"
                #     }
                # ]}])

                # # for loop
                # self.api.for_loop('enumerate', {"iterators": [
                #     "i",
                #     "x"
                # ],
                #     "iterable": "x"
                # })

                # # function call to print
                # self.api.function_call('print', [{'value': 'x'}])
