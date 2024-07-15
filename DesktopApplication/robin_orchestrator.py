import subprocess
import threading

import customtkinter as tk
import tkinterDnD

from speech import *
from subprocess_thread import *
import json
from PIL import Image
import pydirectinput


class RobinOrchestrator:

    def __init__(self,):

        # read config file
        with open('./config.json') as f:
            self.__config = json.load(f)

        # Voice Recognition Model
        self.sr = None
        self.sr_thread = None

        # Mouse tracking thread
        self.mouse_thread = None

        tk.set_ctk_parent_class(tkinterDnD.Tk)

        tk.set_appearance_mode("dark")

        tk.set_default_color_theme("theme/violet.json")

        self.app = tk.CTk()

        self.app.geometry("250x300")
        # self.app.resizable(False, False)

        # icon
        self.app.iconbitmap('./Assets/robin_transparent.ico')

        self.app.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.app.title("Robin")

        self.frame_1 = tk.CTkFrame(master=self.app)

        self.frame_1.pack(pady=0, padx=0, fill="both", expand=True)

        my_image = tk.CTkImage(light_image=Image.open('assets/robin_transparent.png'),
                               dark_image=Image.open(
                                   './Assets/robin_transparent.png'),
                               size=(150, 150))

        my_label = tk.CTkLabel(self.frame_1, text="", image=my_image)
        my_label.pack(pady=0)

        self.active_switch_state = tk.IntVar(value=0)

        self.active_switch = tk.CTkSwitch(master=self.frame_1,
                                          text="Active",
                                          font=(
                                              "Arial", 20, "bold"),
                                          command=self.handle_robin,
                                          variable=self.active_switch_state,
                                          onvalue=1,
                                          offvalue=0
                                          )

        self.active_switch.pack(pady=10, padx=15, anchor='w')

        # mouse variable
        self.mouse_switch_state = tk.IntVar(value=0)
        self.mouse_switch = tk.CTkSwitch(master=self.frame_1,
                                         text="Mouse",
                                         font=(
                                             "Arial", 20, "bold"),
                                         variable=self.mouse_switch_state,
                                         onvalue=1,
                                         offvalue=0,
                                         command=self.handle_mouse_tracking
                                         )

        self.mouse_switch.pack(pady=10, padx=15, anchor='w')
        # interactive mode variable
        self.interactive_switch_state = tk.IntVar(value=0)
        self.interactive_switch = tk.CTkSwitch(master=self.frame_1,
                                               text="Interactive",
                                               font=(
                                                   "Arial", 20, "bold"),
                                               variable=self.interactive_switch_state,
                                               onvalue=1,
                                               offvalue=0,
                                               command=self.handle_interactive
                                               )

        self.interactive_switch.pack(pady=10, padx=15, anchor='w')

        self.app.attributes('-topmost', True)

    def handle_robin(self):
        print(self.active_switch_state.get())
        if self.active_switch_state.get() == 0:
            self.sr_thread.stop_event.set()
        else:
            self.handle_voice_recognition()

    def activate_robin(self):
        if self.active_switch_state.get() == 0:
            self.active_switch.select()
            self.handle_voice_recognition()
            return True

        # self.active_switch.deselect()
        # self.sr_thread.stop_event.set()

        return False

    def handle_voice_recognition(self):

        if self.active_switch_state.get() == 1:
            self.sr = SpeechRecognition(self)
            self.sr_thread = threading.Thread(
                target=self.sr.recognize, args=())
            self.sr_thread.stop_event = threading.Event()
            self.sr_thread.daemon = True
            self.sr_thread.start()
        else:
            self.sr_thread.stop_event.set()

    def handle_mouse_tracking(self):

        # run mouse subprocess on thread
        if self.mouse_switch_state.get() == 1:
            self.mouse_thread = SubprocessThread(
                self.__config['mouse_execution_path'])
            self.mouse_thread.start()

        else:
            self.mouse_thread.stop()

    def right_click_mouse(self):
        pydirectinput.rightClick()

    def left_click_mouse(self):
        pydirectinput.leftClick()

    def deactivate_robin(self):
        if self.active_switch_state.get() == 1:
            self.active_switch.deselect()
            return True
        return False

    def handle_interactive(self):
        if self.interactive_switch_state.get() == 1:
            # self.sr.activate_interactive()
            self.sr_interactive = threading.Thread(
                target=self.sr.activate_interactive, args=())
            self.sr_interactive.stop_event = threading.Event()
            self.sr_interactive.daemon = True
            self.sr_interactive.start()
        else:
            self.sr.deactivate_interactive()
            self.sr_interactive.stop_event.set()
            print("Non Interactive")

    def run(self):
        self.app.mainloop()

    # on closing kill all threads
    def on_closing(self):
        if self.sr_thread is not None:
            self.sr_thread.stop_event.set()
        if self.mouse_thread is not None:
            self.mouse_thread.stop()
        self.app.destroy()
        exit(0)
