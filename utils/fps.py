from collections import deque
import time


class FPS:
    def __init__(self, length=30):

        self._processing_time_list = deque(maxlen=length)
        self._timer = 0

    def set_reference(self):

        self._timer = time.time()

    def get_reference(self):

        return self._timer

    def add_time(self):

        self._processing_time_list.append(time.time()-self._timer)

    def get_fps(self):

        if len(self._processing_time_list) == 0:
            return 0
        return 1/(sum(self._processing_time_list)/len(self._processing_time_list))
