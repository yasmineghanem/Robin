
import argparse
import threading

import openwakeword

from gui import *
from wake_word import *

if __name__ == "__main__":
    openwakeword.utils.download_models()
    # start_reloader()
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", help="How much audio (in number of samples) to predict on at once",
                        type=int, default=1280, required=False)
    parser.add_argument("--model_path", help="The path of a specific model to load",
                        type=str, default="./Assets/wake_word_model/hey_robin_2.onnx", required=False)
    parser.add_argument("--inference_framework", help="The inference framework to use (either 'onnx' or 'tflite'",
                        type=str, default='onnx', required=False)

    args = parser.parse_args()

    gui = GUI()

    t = threading.Thread(target=wake_word_detection, args=(args, gui))
    t.stop_event = threading.Event()
    t.daemon = True
    t.start()
    gui.run()
