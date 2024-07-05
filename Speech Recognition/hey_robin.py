# import pyaudio
# import numpy as np
# from openwakeword.model import Model

# # Initialize your trained model
# model = Model(wakeword_models=["./Hey_Robin.tflite"],  # can also leave this argument empty to load all of the included pre-trained models
#               )

# # Audio stream configuration
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000  # Ensure this matches your model's requirements
# CHUNK = 1024

# # Initialize PyAudio
# audio = pyaudio.PyAudio()

# # Open audio stream
# stream = audio.open(format=FORMAT, channels=CHANNELS,
#                     rate=RATE, input=True, frames_per_buffer=CHUNK)

# print("Listening for wake word...")

# try:
#     while True:
#         # Read a chunk of data from the audio stream
#         data = stream.read(CHUNK)

#         # Convert audio data to numpy array
#         audio_data = np.frombuffer(data, dtype=np.int16)

#         # Preprocess the audio data if needed (e.g., normalization, feature extraction)
#         # preprocessed_audio_data = preprocess(audio_data)

#         # Use your model to detect the wake word
#         if model.predict(audio_data):  # Replace with the actual method to predict
#             print("Wake word detected!")
#             # Trigger your action here
#             break  # Exit the loop or handle the wake word detection as needed

# except KeyboardInterrupt:
#     print("Interrupted by user")

# finally:
#     # Close the audio stream
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()

# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Imports
import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse
import openwakeword

# One-time download of all pre-trained models (or only select models)
# openwakeword.utils.download_models()


# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once",
    type=int,
    default=1280,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="./hey_robin_2.tflite",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default='tflite',
    required=False
)

args = parser.parse_args()
print(args)

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load pre-trained openwakeword models
if args.model_path != "":
    owwModel = Model(wakeword_models=[
                     args.model_path], inference_framework=args.inference_framework)
else:
    owwModel = Model(inference_framework=args.inference_framework)

n_models = len(owwModel.models.keys())

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    # Generate output string header
    print("\n\n")
    print("#"*100)
    print("Listening for wakewords...")
    print("#"*100)
    print("\n"*(n_models*3))

    while True:
        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

        # Column titles
        n_spaces = 16
        output_string_header = """
            Model Name         | Score | Wakeword Status
            --------------------------------------
            """

        for mdl in owwModel.prediction_buffer.keys():
            # Add scores in formatted table
            scores = list(owwModel.prediction_buffer[mdl])
            curr_score = format(scores[-1], '.20f').replace("-", "")

            output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
            """

        # Print results table
        print("\033[F"*(4*n_models+1))
        print(output_string_header, "                             ", end='\r')
        if list(owwModel.prediction_buffer[mdl])[-1] > 0.7:
            break
