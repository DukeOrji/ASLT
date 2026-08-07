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

        self.prediction = result["prediction"]
        self.second_prediction = result["second_prediction"]

        
    def send_label(self, letter):
        if letter == "space":
            letter = " "
        self.current_word += letter
        self.word_box.setText(self.current_word)

    def remove_last_label(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        self.word_box.setText(self.current_word)

    def __init__(self):
        super().__init__()

        self.current_word = ''
        self.current_alt_word = ''

        with open("style.qss", 'r') as f:
                self.setStyleSheet(f.read())

        #ACCURATE PREDICTION BUTTON
        self.prediction_btn = QPushButton()
        pred_layout = QVBoxLayout(self.prediction_btn)

        actual_lab = QLabel('ACTUAL')
        actual_lab.setAlignment(Qt.AlignCenter)
        actual_lab.setProperty("class", "heading_true")

        self.current_letter = QLabel("")
        self.current_letter.setProperty("class", "label")
        self.current_letter.setAlignment(Qt.AlignCenter)

        self.confidence = QLabel(" -- ")
        self.confidence.setProperty("class", "confidence")
        self.confidence.setAlignment(Qt.AlignCenter)

        pred_layout.addWidget(actual_lab)
        pred_layout.addWidget(self.current_letter)
        pred_layout.addWidget(self.confidence)
        pred_layout.setSpacing(15)
        pred_layout.setContentsMargins(15, 15, 15, 15)

        self.prediction_btn.clicked.connect(lambda: self.send_label(self.prediction))
        self.prediction_btn.setFixedHeight(130)

        #ALTERNATIVE PREDICTION BUTTON
        self.alt_btn = QPushButton()
        alt_layout = QVBoxLayout(self.alt_btn)

        alt_lab = QLabel('ALTERNATE')
        alt_lab.setAlignment(Qt.AlignCenter)
        alt_lab.setProperty("class", "heading")

        self.unc_letter = QLabel("")
        self.unc_letter.setProperty("class", "label")
        self.unc_letter.setAlignment(Qt.AlignCenter)

        self.unc_confidence = QLabel(" -- ")
        self.unc_confidence.setProperty("class", "confidence")
        self.unc_confidence.setAlignment(Qt.AlignCenter)

        alt_layout.addWidget(alt_lab)
        alt_layout.addWidget(self.unc_letter)
        alt_layout.addWidget(self.unc_confidence)
        alt_layout.setSpacing(15)
        alt_layout.setContentsMargins(15, 15, 15, 15)
        self.alt_btn.clicked.connect(lambda: self.send_label(self.second_prediction))
        self.alt_btn.setFixedHeight(130)

        #word output
        word_group = QGroupBox('CURRENT WORD')
        word_group.setAlignment(Qt.AlignCenter)
        word_group.setStyleSheet("""
        QGroupBox {
        color: green;
        font-size: 13px;
        font-weight: bold;
        }
        """)
        word_group.setFixedHeight(130)

        word_layout = QVBoxLayout()
        word_group.setLayout(word_layout)
        
        self.word_box = QLineEdit()
        #self.word_box.setFixedheight(50)
        self.word_box.setReadOnly(True)

        word_layout.addWidget(self.word_box)

        #CONSTANT PROPORTION
        self.prediction_btn.setFixedSize(180, 170)
        self.alt_btn.setFixedSize(180, 170)
        word_group.setFixedSize(180, 130)



        #OPEN CAMERA
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        
        #UPDATE FRAME EVERY 30ms
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        #INITIALIZE SERVER
        self.server = self.get_server()

        self.setWindowTitle("ASL Translator")

        main_container = QWidget()
        self.setCentralWidget(main_container)

        main_layout = QHBoxLayout(main_container)
        webcam_layout = QVBoxLayout()
        side_layout = QVBoxLayout()
        

        #space/margin setting
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(15, 15, 15, 15)

        webcam_layout.setSpacing(15)
        side_layout.setSpacing(3)
        
        #WEBCAM SETTINGS
        self.webcam = QLabel()
        self.webcam.setProperty("class", "webcam")
        
        self.webcam.setFixedSize(600, 400)
        self.webcam.setAlignment(Qt.AlignCenter)

        #CAMERA ICON SETTINGS
        confirm_btn = QPushButton()
        confirm_btn.clicked.connect(self.capture_current_frame)
        confirm_btn.setIcon(QIcon("capture.png"))

        #LAYOUT SETTINGS
        main_layout.addLayout(webcam_layout, 3)
        main_layout.addLayout(side_layout, 1)

        #PRIMARY
        webcam_layout.addWidget(self.webcam, alignment=Qt.AlignCenter)
        webcam_layout.addWidget(confirm_btn, alignment=Qt.AlignCenter)
        

        #SECONDARY
        side_layout.addWidget(self.prediction_btn, alignment=Qt.AlignTop)
        side_layout.addWidget(self.alt_btn, alignment=Qt.AlignTop)
        side_layout.addWidget(word_group, alignment=Qt.AlignTop)
        side_layout.addStretch()
        


    
app = QApplication(sys.argv) 

window = MainWindow()
window.show()

app.exec()