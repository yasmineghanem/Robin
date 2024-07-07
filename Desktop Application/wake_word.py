
import pyaudio
import numpy as np
from openwakeword.model import Model
import time
import pyttsx3
from robin_responses import responses


last_detection_time = 0  # Time of the last detection
DEBOUNCE_TIME = 2  # Debounce time in seconds


def wake_word_detection(args, gui):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = args.chunk_size
    audio = pyaudio.PyAudio()
    mic_stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, frames_per_buffer=CHUNK)

    if args.model_path != "":
        owwModel = Model(wakeword_models=[
                         args.model_path], inference_framework=args.inference_framework)
    else:
        owwModel = Model(inference_framework=args.inference_framework)

    n_models = len(owwModel.models.keys())

    while True:
        global last_detection_time

        audio_data = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
        prediction = owwModel.predict(audio_data)

        current_time = time.time()
        if prediction['hey_robin_2'] > 0.7 and (current_time - last_detection_time) > DEBOUNCE_TIME:
            print("Wake word detected!")
            last_detection_time = current_time
            try:
                # check if it's not activated
                if gui.activate_robin():
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
                # else: 
                #     gui.stop_voice_recognition()
                #     print('deactivate')
                    # gui.
            except Exception as e:
                print(f"Error in app.after: {e}")
