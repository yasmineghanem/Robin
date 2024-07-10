import customtkinter as tk
import tkinterDnD
import threading
from speech import *


class GUI:

    def __init__(self):
        
        self.sr = SpeechRecognition(self)
        
        self.sr_thread = threading.Thread(target=self.sr.recognize)
        self.sr_thread.daemon = True

        tk.set_ctk_parent_class(tkinterDnD.Tk)

        tk.set_appearance_mode("dark")

        tk.set_default_color_theme("theme/violet.json")

        self.app = tk.CTk()

        self.app.geometry("400x200")

        self.app.title("Robin")

        self.frame_1 = tk.CTkFrame(master=self.app)

        self.frame_1.pack(pady=0, padx=0, fill="both", expand=True)

        self.active_switch_state = tk.IntVar(value=0)

        self.active_switch = tk.CTkSwitch(master=self.frame_1,
                                          text="Active",
                                          font=(
                                              "Arial", 20, "bold"),
                                          #  command=self.active_switch_state_changed,
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
                                         #  command=self.active_switch_state_changed,
                                         variable=self.mouse_switch_state,
                                         onvalue=1,
                                         offvalue=0
                                         )

        self.mouse_switch.pack(pady=10, padx=10)

        # # button to toggle switch

        # self.toggle_button = tk.CTkButton(

        #     master=self.frame_1, text="Toggle", command=self.toggle)

        # self.toggle_button.pack(pady=10, padx=10)

        self.app.attributes('-topmost', True)

    def active_switch_state_changed(self):

        # change the variable's value
        self.active_switch_state.set(not self.active_switch_state.get())

    def toggle(self):

        if self.active_switch_state.get() == 1:
            self.active_switch.deselect()
        else:
            self.active_switch.select()

    def activate_robin(self):
        if self.active_switch_state.get() == 0:
            self.active_switch.select()
            # s = SpeechRecognition()
            # start speech recognition on different thread
            # t = threading.Thread(target=self.sr.recognize)
            # t.daemon = True
            self.sr_thread.start()
            return True
        return False
    
    # def stop_voice_recognition(self):
    #     # self.sr.deactivate()
    #     print('aho')
    #     self.sr_thread.join()
        
        
    
    def deactivate_robin(self):
        if self.active_switch_state.get() == 1:
            self.active_switch.deselect()
            return True
        return False

    def run(self):
        self.app.mainloop()
