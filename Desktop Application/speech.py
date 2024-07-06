
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import pyaudio
import requests


class SpeechRecognition:
    def __init__(self):
        # self.recognizer = sr.Recognizer()
        # self.microphone = sr.Microphone()

        # vosk
        self.model = Model("./Assets/voice_recognition_models/v_2")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.mic = pyaudio.PyAudio()
        
        
        self.stream = self.mic.open(format=pyaudio.paInt16, channels=1,
                                    rate=16000, input=True, frames_per_buffer=8192)
        self.active = True

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
                print(self.recognizer.Result())

                # request to localhost:2805
                # try:
                #     response = requests.get('http://localhost:2805/git/pull')
                #     print(response.text)
                    
                # except Exception as e:
                #     print(f"Error in recognize: {e}")
                    

    # destructor

    def __del__(self):
        self.stream.stop_stream()
        self.stream.close()
        self.mic.terminate()
        print('SpeechRecognition object deleted')
