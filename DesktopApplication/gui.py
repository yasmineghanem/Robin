import subprocess
import threading

import customtkinter as tk
import tkinterDnD

from speech import *
from stoppable_thread import *
from subprocess_thread import *
from wake_word import *


class GUI:

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

        self.app.geometry("400x200")

        self.app.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.app.title("Robin")

        self.frame_1 = tk.CTkFrame(master=self.app)

        self.frame_1.pack(pady=0, padx=0, fill="both", expand=True)

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

        self.active_switch.pack(pady=10, padx=10)

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

        self.mouse_switch.pack(pady=10, padx=10)

        self.app.attributes('-topmost', True)

    def handle_robin(self):
        if self.active_switch_state.get() == 0:
            self.sr_thread.stop_event.set()

    def activate_robin(self):
        if self.active_switch_state.get() == 0:
            self.active_switch.select()
            self.handle_voice_recognition()
            return True
        return False

    # def handle_activate_robin(self):

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

    def deactivate_robin(self):
        if self.active_switch_state.get() == 1:
            self.active_switch.deselect()
            return True
        return False

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
