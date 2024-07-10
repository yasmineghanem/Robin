
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import pyaudio
import requests
from api import *
from CommandIntent.final_files.command_intent import CommandIntent

class SpeechRecognition:
    def __init__(self, gui):
        # self.recognizer = sr.Recognizer()
        # self.microphone = sr.Microphone()
        self.gui = gui
        self.command_intent = CommandIntent('../CommandIntent/models/intent_detection_model.keras',
                                            '../CommandIntent/models/ner_model2.pth')

        # vosk
        self.model = Model("./Assets/voice_recognition_models/v_2")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.mic = pyaudio.PyAudio()

        self.stream = self.mic.open(format=pyaudio.paInt16, channels=1,
                                    rate=16000, input=True, frames_per_buffer=8192)
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

                # response = self.command_intent.process_command(r['text'])
                response = self.command_intent.process_command('cast variable x to integer')

                

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
