from collections import deque
from threading import Thread
import cv2


class VideoStream:
    """
    code adapted from:
    https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv
    Obs: Added the deque from collections for performance gains
    """
    def __init__(self, path, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path, cv2.CAP_GSTREAMER)
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.Q = deque(maxlen=queue_size)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return

            # read the next frame from the file
            (grabbed, frame) = self.stream.read()

            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed:
                self.stop()
                return
            # add the frame to the queue
            self.Q.append(frame)

    def read(self):
        # return next frame in the queue
        return self.Q.pop()

    def more(self):
        # return True if there are still frames in the queue
        return len(self.Q) > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
