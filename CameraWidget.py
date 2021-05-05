import cv2
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage

class CameraWidget(QWidget):
    send_video = pyqtSignal(QImage)
    
    def __init__(self):
        super(CameraWidget, self).__init__()
        self.camera = 0
        
        
    @pyqtSlot()
    def startCapturing(self):
        self.camera = cv2.VideoCapture(0)  
        while self.camera.isOpened():
            _, image = self.camera.read()
            self.displayImage(image)
            cv2.waitKey()

        self.camera.release()
        cv2.destroyAllWindows()
        
    def displayImage(self, img):
        qformat = QImage.Format_Indexed8 # Find correct image format
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], qformat) # Create QImage object
        img = img.rgbSwapped()
        self.send_video.emit(img) # Send the img object to the main application
    