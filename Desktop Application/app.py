import customtkinter
import tkinterDnD
import hupper
import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse
import openwakeword
import threading
import time


# def start_reloader():
#     reloader = hupper.start_reloader('app.gui')


def main():
    customtkinter.set_ctk_parent_class(tkinterDnD.Tk)

    customtkinter.set_appearance_mode("dark")

    customtkinter.set_default_color_theme("theme/violet.json")

    app = customtkinter.CTk()
    app.geometry("400x200")
    app.title("Robin")

    # app.iconbitmap("./Assets/robin.png")
    print(type(app), isinstance(app, tkinterDnD.Tk))

    # def slider_callback(value):
    #     progressbar_1.set(value)

    frame_1 = customtkinter.CTkFrame(master=app)
    frame_1.pack(pady=0, padx=0, fill="both", expand=True)

    switch_1 = customtkinter.CTkSwitch(master=frame_1,
                                       text="Active",
                                       command=switch_state_changed,
                                       )
    switch_1.pack(pady=10, padx=10)

    app.attributes('-topmost', True)

    app.mainloop()


class GUI:
    def __init__(self):
        self.app = customtkinter.CTk()
        self.app.geometry("400x200")
        self.app.title("Robin")

        self.frame_1 = customtkinter.CTkFrame(master=self.app)
        self.frame_1.pack(pady=0, padx=0, fill="both", expand=True)

        self.switch_state = customtkinter.IntVar(value=0)

        self.switch_1 = customtkinter.CTkSwitch(master=self.frame_1,
                                                text="Active",
                                                command=self.switch_state_changed,
                                                variable=self.switch_state,
                                                onvalue=1,
                                                offvalue=0
                                                )
        self.switch_1.pack(pady=10, padx=10)

        # button to toggle switch
        self.toggle_button = customtkinter.CTkButton(
            master=self.frame_1, text="Toggle", command=self.toggle)
        self.toggle_button.pack(pady=10, padx=10)

        self.app.attributes('-topmost', True)

    def switch_state_changed(self):
        # if self.switch_state.get() == 1:
        #     print("Switch is ON")
        # else:
        #     print("Switch is OFF")

        # change the variable's value
        self.switch_state.set(not self.switch_state.get())
        print(self.switch_state.get())

    def toggle(self):
        if self.switch_state.get() == 1:
            self.switch_1.deselect()
        else:
            self.switch_1.select()

    def run(self):
        self.app.mainloop()


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
        audio_data = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
        prediction = owwModel.predict(audio_data)

        n_spaces = 16
        output_string_header = """
        Model Name         | Score | Wakeword Status
        --------------------------------------
        """

        for mdl in owwModel.prediction_buffer.keys():
            scores = list(owwModel.prediction_buffer[mdl])
            curr_score = format(scores[-1], '.20f').replace("-", "")
            output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
            """

        print("\033[F"*(4*n_models+1))
        print(output_string_header, "                             ", end='\r')
        if list(owwModel.prediction_buffer[mdl])[-1] > 0.7:
            print("Wakeword detected!")
            gui.toggle()
            # Perform additional actions upon wakeword detection


if __name__ == "__main__":
    # start_reloader()
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", help="How much audio (in number of samples) to predict on at once",
                        type=int, default=1280, required=False)
    parser.add_argument("--model_path", help="The path of a specific model to load",
                        type=str, default="./hey_robin_2.onnx", required=False)
    parser.add_argument("--inference_framework", help="The inference framework to use (either 'onnx' or 'tflite'",
                        type=str, default='tflite', required=False)

    args = parser.parse_args()

    gui = GUI()

    t = threading.Thread(target=wake_word_detection, args=(args, gui))
    t.daemon = True
    t.start()
    gui.run()
