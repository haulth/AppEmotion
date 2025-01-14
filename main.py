import sys
import cv2
import numpy as np
from loadModel import Classifier
from time import time
import facenet
from PIL import Image
import align.detect_face
import tensorflow as tf
############################
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject,QThread
from Display import Ui_MainWindow
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog
###############################
from tensorflow.keras.models import Model, load_model

tf.compat.v1.disable_eager_execution()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
with sess.as_default():
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")

classifier = Classifier()
classifier.load_model()

INPUT_IMAGE_SIZE = 96
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709


class MainWindown(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = QWidget()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.uic.ButtonStart.clicked.connect(self.start_capture_video)
        self.uic.ButtonStop.clicked.connect(self.stop_capture_video)
        self.thread = {}
    def closeEnvent(self, event):
        self.stop_capture_video()
    def start_capture_video(self):
        self.thread[1] = live_stream(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)
    def stop_capture_video(self):
        self.thread[1].stop()
    def show_wedcam(self, frame):
        qt_img = self.convert_cv_qt(frame)
        self.uic.label.setPixmap(qt_img)
    def convert_cv_qt(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch*w
        convert_to_Qt_format = QtGui.QImage(rgb_img.data, w,h,bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
class live_stream(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(live_stream, self).__init__()
    def run(self):
        self.run_programer()
    def predict_model(self, frame):
        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        faces_found = bounding_boxes.shape[0]
        try:
            if faces_found > 1:
                cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (255, 255, 255), thickness=1, lineType=2)
            elif faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)
                for i in range(faces_found):
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]
                    print(bb[i][3] - bb[i][1])
                    print(frame.shape[0])
                    print((bb[i][3] - bb[i][1]) / frame.shape[0])
                    if ((bb[i][3] - bb[i][1]) / frame.shape[0]) > 0.25:
                        print("Emotion detected")
                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                            interpolation=cv2.INTER_CUBIC)
                        cv2.imshow('Face ', scaled)
                        # put rectangle to main image
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                        name = classifier.predict(scaled)
                        # put name
                        cv2.putText(frame, name, (bb[i][0], bb[i][1] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                        print("Name: {}".format(name))

        except:
            pass
        return frame
    def get_camera_stream(self):
        return cv2.VideoCapture(0)
    def run_programer(self):
        vd = self.get_camera_stream()
        assert vd.isOpened()
        total_fps = 0
        frame_count = 0
        while True:
            start_time = time()
            ret, frame = vd.read()
            assert ret
            frame = self.predict_model(frame)
            end_time = time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            print(f"Frame Per Second: {round(fps, 1)}FPS")
            self.signal.emit(frame)
    def stop(self):
        print("stop threading", self.index)
        self.terminate()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindown()
    main_win.show()
    sys.exit(app.exec())