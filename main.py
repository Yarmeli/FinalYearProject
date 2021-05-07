# main.py
import sys, os
from PyQt5 import QtCore, uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog


import CameraWidget, ThumbWidget

from helpers import Debug

# Find the correct path of the .ui file
ui_path = os.path.dirname(os.path.abspath(__file__))
form_class = uic.loadUiType(os.path.join(ui_path, "MainWindow.ui"))[0]

class MainWindow(QMainWindow, form_class):
    closeCamera = pyqtSignal()
    def __init__(self):
        super().__init__()
        # Load the UI
        uic.loadUi("MainWindow.ui", self)
        
        self.camWidget = CameraWidget.CameraWidget()
        self.camWidget.send_msg.connect(self.setOutputText)
        self.camWidget.send_video.connect(self.setVideoFeed)
        self.startCamera.clicked.connect(self.camWidget.startCapturing)
        self.closeCamera.connect(self.camWidget.closeCameraIfOpened)
        
        self.takePicture.clicked.connect(self.camWidget.savePicture)
        self.uploadPicture.clicked.connect(self.uploadFiles)
        
        
        self.thumbWidget = ThumbWidget.ThumbWidget()
        self.actionThumb_Settings.triggered.connect(self.thumbWidget.show)
        self.actionClose.triggered.connect(self.close)
        
        self.setOutputText("Please choose 'Start Camera' or 'Upload Pictures'")
        self.setOutputText("You can change your thumb settings on the top left 'File' dropdown\n")
        
        
    @pyqtSlot(str)
    def setOutputText(self, text):
        Debug("Output", text)
        self.output_box.append(text)    
    
    @pyqtSlot(QImage)
    def setVideoFeed(self, img):
        self.videoFeed.setPixmap(QPixmap.fromImage(img))
        self.videoFeed.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    
    def uploadFiles(self):
        fileNames, _ = QFileDialog.getOpenFileNames(self,
            "Open Images", "", "Image Files (*.png *.jpg *.bmp)");
        if fileNames:
            print(fileNames)
            """
                Pending work:
                    Replicate same behaviour of camWidget
            """
    
    
    def closeEvent(self, event):
        self.closeCamera.emit() # Close camera if needed
        event.accept() # let the window close
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec_()
