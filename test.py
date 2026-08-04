import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QGridLayout, QPushButton, QLineEdit, QTextEdit, QComboBox, QListWidget, QSlider
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QIcon
import cv2
from server import Server
from mpipeline import get_landmarks
import torch
import os



class MainWindow(QMainWindow):


    def __init__(self):
        super().__init__()

        self.setWindowTitle('Test')
        self.setStyleSheet("""
        QMainWindow {
            background-color: #202124;
        }
        """)
        
        container = QWidget()
        self.setCentralWidget(container)
        container.setStyleSheet("""
        background-color:  #202123; 
        """)

        layout = QGridLayout(container)

        label = QLabel("good job")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
        QLabel {
            color: white;
            font-size: 24px;
            font-weight: bold;
        }
        """)

        btn = QPushButton('DO IT')
        btn.clicked.connect(lambda: print('okay oo'))
        btn.setIcon(QIcon("btn.png"))
        btn.setStyleSheet("""
        QPushButton {
            background-color: #2563EB;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        }

        QPushButton:hover {
            background-color: #1D4ED8;
        }

        QPushButton:pressed {
            background-color: #1E40AF;
        }
        """)

        line = QLineEdit()
        line.setReadOnly(True)
        line.setStyleSheet("""
        QLineEdit {
            background-color: #2B2B2B;
            color: white;
            border: 2px solid #555;
            border-radius: 8px;
            padding: 8px;
            font-size: 18px;
        }
        """)

        layout.addWidget(label, 0, 0)
        layout.addWidget(btn, 1, 0)
        layout.addWidget(line, 2, 0)
        

    

    
app = QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec()