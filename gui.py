import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QGridLayout, QPushButton, QLineEdit, QTextEdit, QComboBox, QListWidget, QSlider
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
from server import Server
from mpipeline import get_landmarks
import torch
import os
from config import device


class MainWindow(QMainWindow):

    def get_server(self):
        server = Server(
            train_set=None,
            test_set=None
        )

        if os.path.exists("checkpoint.pth"):
            checkpoint = torch.load("checkpoint.pth", map_location=device)

            optim_dict = checkpoint["optimizer_state_dict"]
            model_dict = checkpoint["model_state_dict"]
            server.set_weight(model_dict, optim_dict)

        return server
    
    def closeEvent(self, event):
        self.cap.release() #close the webcam before ending application
        event.accept()

    #init backspace key
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace:
            self.remove_last_label()

    def update_frame(self):
        
        success, frame = self.cap.read()
        if not success:
            return
        
        self.frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        h, w, ch = frame.shape

        #convert to align with Qt's features
        image = QImage(
            frame.data,
            w,
            h,
            ch * w,
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.webcam.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.webcam.setPixmap(pixmap)

    def capture_current_frame(self):

        if not hasattr(self, "frame"):
            return
        
        landmarks = get_landmarks(self.frame)
        if landmarks is None:
            return

        data = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
        result = self.server.predict_label(data)
        self.current_letter.setText(
            f"Current Letter: {result['prediction']}"
        )

        self.unc_letter.setText(
            f"Maybe: {result['second_prediction']}"
        )

        self.confidence.setText(
            f"Confidence: {result['confidence']:.1%}"
        )

        self.unc_confidence.setText(
            f"Confidence: {result['second_confidence']:.1%}"
        )

        label = result["prediction"]
        self.current_word += label
        self.word_box.setText(self.current_word)

    def remove_last_label(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        self.word_box.setText(self.current_word)

    def __init__(self):
        super().__init__()

        self.current_letter = QLabel("Current Letter:")
        self.unc_letter = QLabel("Maybe:")
        self.confidence = QLabel("Confidence: --%")
        self.unc_confidence = QLabel("Confidence: --%")
        self.current_word = ''
        
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) #update frame every 30ms

        #initialize server and run prediction
        self.server = self.get_server()

        self.setWindowTitle("ASL Translator")

        webcam_container = QWidget()
        primary_container = QWidget()

        self.setCentralWidget(webcam_container)

        webcam_layout = QGridLayout(webcam_container)
        primary_layout = QGridLayout(primary_container)
        

        self.webcam = QLabel()
        self.webcam.setFixedSize(600, 400)
        self.webcam.setAlignment(Qt.AlignCenter)


        
        current_wl = QLabel('Current Word:')
        self.word_box = QLineEdit()
        self.word_box.setReadOnly(True)

        confirm_btn = QPushButton('Confirm')
        confirm_btn.clicked.connect(self.capture_current_frame)


        webcam_layout.addWidget(self.webcam, 0, 0)
        webcam_layout.addWidget(primary_container, 0, 1)
        primary_layout.addWidget(self.current_letter, 1, 0)
        primary_layout.addWidget(self.unc_letter, 1, 1)
        primary_layout.addWidget(self.confidence, 2, 0)
        primary_layout.addWidget(self.unc_confidence, 2, 1)
        primary_layout.addWidget(current_wl, 3, 0)
        primary_layout.addWidget(self.word_box, 4, 0)
        webcam_layout.addWidget(confirm_btn, 1, 0)

    
app = QApplication(sys.argv) 

window = MainWindow()
window.show()

app.exec()