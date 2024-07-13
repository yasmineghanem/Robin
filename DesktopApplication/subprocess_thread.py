import subprocess
import threading
import os


DEVNULL = open(os.devnull, 'wb')


class SubprocessThread(threading.Thread):
    def __init__(self, exe_path):
        super().__init__()
        self.exe_path = exe_path
        self.process = None

        self.stop_signal = threading.Event()
        self.daemon = True  # Set the thread as a daemon thread

    def run(self):
        try:
            self.process = subprocess.Popen(
                [self.exe_path],  shell=False, stdout=DEVNULL, stderr=DEVNULL)

        except Exception as e:
            print(f"Error running subprocess: {e}")

    def stop(self):
        self.stop_signal.set()
        if self.process is not None:
            self.process.terminate()
