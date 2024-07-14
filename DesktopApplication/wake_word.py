import pyaudio
import numpy as np
from openwakeword.model import Model
# import time
import pyttsx3
from robin_responses import responses
from datetime import datetime, timedelta
import threading

# now - 3 seconds
last_detection_time = datetime.now() - timedelta(seconds=3)
DEBOUNCE_TIME = 5  # Debounce time in seconds


def wake_word_detection(args, ro):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = args.chunk_size
    audio = pyaudio.PyAudio()
    first_time = True
    mic_stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, frames_per_buffer=CHUNK)

    if args.model_path != "":
        owwModel = Model(wakeword_models=[
                         args.model_path], inference_framework=args.inference_framework)
    else:
        owwModel = Model(inference_framework=args.inference_framework)

    n_models = len(owwModel.models.keys())
    # print('alooo')
    while True:
        global last_detection_time

        audio_data = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
        prediction = owwModel.predict(audio_data)

        current_time = datetime.now()
        print((current_time - last_detection_time).total_seconds())

        if prediction['hey_robin_2'] > 0.7 and ((current_time - last_detection_time).total_seconds() > DEBOUNCE_TIME):
            # flush buffer
            mic_stream.read(mic_stream.get_read_available())
            print("Wake word detected!")
            try:
                # check if it's not activated
                # if first_time:
                #     first_time = False
                # ro.activate_robin()
                t = threading.Thread(
                    target=ro.activate_robin, args=())
                t.daemon = True
                t.start()
                last_detection_time = current_time

                # activate it
                engine = pyttsx3.init()

                # getting details of current voice
                voices = engine.getProperty("voices")

                engine.setProperty(
                    "voice", voices[1].id
                )  # changing index, changes voices. 1 for female

                # random response
                engine.say(responses[np.random.randint(0, len(responses))])
                engine.runAndWait()

            except Exception as e:
                print(f"Error in app.after: {e}")
