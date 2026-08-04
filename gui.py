import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QGroupBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QIcon
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
            f"{result['prediction']}"
        )

        self.unc_letter.setText(
            f"{result['second_prediction']}"
        )

        self.confidence.setText(
            f"{result['confidence']:.1%}"
        )

        self.unc_confidence.setText(
            f"{result['second_confidence']:.1%}"
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
        with open("style.qss", 'r') as f:
                self.setStyleSheet(f.read())


        pred_box = QGroupBox('ACTUAL')
        alt_box = QGroupBox('MAYBE')
        word_group = QGroupBox('CURRENT WORD')

        pred_layout = QVBoxLayout()
        alt_layout = QVBoxLayout()
        word_layout = QVBoxLayout()

        pred_box.setLayout(pred_layout)
        alt_box.setLayout(alt_layout)
        word_group.setLayout(word_layout)


        self.current_letter = QLabel("")
        self.unc_letter = QLabel("")
        self.confidence = QLabel(" -- ")
        self.unc_confidence = QLabel(" -- ")
        self.current_word = ''

        self.unc_confidence.setStyleSheet("""
        QLabel {
            background-color: #8c898f;
            font-size: 15px;
            font-weight: bold;
            color: black;
        }
        """)

        self.confidence.setStyleSheet("""
        QLabel {
            background-color: #8c898f;
            font-size: 15px;
            font-weight: bold;
            color: black;
        }
        """)


        
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) #update frame every 30ms

        #initialize server and run prediction
        self.server = self.get_server()

        self.setWindowTitle("ASL Translator")

        main_container = QWidget()
        self.setCentralWidget(main_container)

        main_layout = QHBoxLayout(main_container)
        webcam_layout = QVBoxLayout()
        side_layout = QVBoxLayout()
        

        #space/margin setting
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(20, 20, 20, 20)

        webcam_layout.setSpacing(15)
        side_layout.setSpacing(3)
        

        self.webcam = QLabel()
        self.webcam.setStyleSheet("""
        QLabel {
            background-color: #8c898f;
        }
        """)
        self.webcam.setFixedSize(600, 400)
        self.webcam.setAlignment(Qt.AlignCenter)


        current_letter_lab = QLabel('ACTUAL')
        current_letter_lab.setStyleSheet("""
        QLabel {
            background-color: #8c898f;
            color: cyan;
            font-size: 10px;
        }
        """)

        unc_letter_lab = QLabel('MAYBE')
        unc_letter_lab.setStyleSheet("""
        QLabel {
            background-color: #8c898f;
            color: red;
            font-size: 10px;
        }
        """)

        
        current_wl = QLabel('CURRENT WORD')
        current_wl.setStyleSheet("""
        QLabel {
            background-color: #8c898f;
            font-size: 13px;
            font-weight: bold;
            color: black;
        }
        """)

        self.word_box = QLineEdit()
        self.word_box.setReadOnly(True)

        confirm_btn = QPushButton()
        confirm_btn.clicked.connect(self.capture_current_frame)
        confirm_btn.setIcon(QIcon("capture.png"))

        main_layout.addLayout(webcam_layout, 3)
        main_layout.addLayout(side_layout, 1)

        webcam_layout.addWidget(self.webcam, alignment=Qt.AlignCenter)
        webcam_layout.addWidget(confirm_btn, alignment=Qt.AlignCenter)

        pred_layout.addWidget(self.current_letter)
        pred_layout.addWidget(self.confidence)
        alt_layout.addWidget(self.unc_letter)
        alt_layout.addWidget(self.unc_confidence)
        word_layout.addWidget(self.word_box)

        side_layout.addWidget(pred_box)
        side_layout.addWidget(alt_box)
        side_layout.addWidget(word_group)
        side_layout.addStretch()


    
app = QApplication(sys.argv) 

window = MainWindow()
window.show()

app.exec()