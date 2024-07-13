'''
Extending the thread class to be stoppable with a thread event
'''

import threading
import time


class StoppableThread(threading.Thread):
    def __init__(self, target, args):
        super().__init__(target=target, args=args)
        self.stop_event = threading.Event()

        # daemon
        self.daemon = True

    def run(self):
        self.start()

    def stop(self):
        self.stop_event.set()
