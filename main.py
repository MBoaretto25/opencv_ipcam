# -*- coding: utf-8 -*-
import time
import cv2

from .utils import VideoStream
from .utils import NNModel
from .utils import FPS


def main():

    print("[INFO] starting communication with ip camera...")
    rtspsrc = {
        "user": user,
        "password": password,
        "ip": ip,
        "port": port,
        "profile": profile
    }
    rtsp_url = "rtsp://{}:{}@{}:{}/{}".format(
        rtspsrc['user'],
        rtspsrc['password'],
        rtspsrc['ip'],
        rtspsrc['port'],
        rtspsrc['profile'],
    )
    gst_pipeline = "rtspsrc location={} latency=0 ! decodebin ! videoconvert ! appsink".format(rtsp_url)

    print("[INFO] starting video thread...")
    # Initialize video streaming
    stream = VideoStream(gst_pipeline).start()
    time.sleep(1.0)

    fps = FPS()

    print("[INFO] initializing neural net model...")
    # Initialize NN Model
    nnet_params = {
        "frozen_graph" : "frozen_graph",
        "input_node" : "input_node",
        "output_node" : "output_node",
    }
    nnet = NNModel(nnet_params)

    # Initialize Windows with OpenGL rendering
    cv2.namedWindow('IpCam', cv2.WND_PROP_OPENGL)
    cv2.resizeWindow('IpCam', 640, 480)

    start = time.time()
    time_limit = start + 60  # run for a minute
    while True:

        if time.time() > time_limit:
            print("[INFO] time limit reached...")
            break

        fps.set_reference()
        img = stream.read()

        # Post Process frames
        img = cv2.resize(img, (240, 240))

        # Perform predictions
        predicted_img = nnet.predict(img)

        cv2.putText(predicted_img, str(int(fps.get_fps())), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        # img.copy is important here for OpenGl rendering
        cv2.imshow('IpCam', predicted_img.copy())
        cv2.waitKey(1)

        fps.add_time()

    cv2.destroyAllWindows()
    stream.stop()
    print("[INFO] done!")


if __name__ == "__main__":
    main()
