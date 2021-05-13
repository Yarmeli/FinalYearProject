import cv2
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage

from model import GetPrediction

class CameraWidget(QWidget):
    send_msg = pyqtSignal(str)
    send_video = pyqtSignal(QImage)
    send_prediction = pyqtSignal(int)
    
    def __init__(self):
        super(CameraWidget, self).__init__()
        self.camera = 0
        self.saveCurrentFrame = 0
        
        
    @pyqtSlot()
    def startCapturing(self):
        self.send_msg.emit('Starting WebCam....\n')
        self.send_msg.emit('You need to take one picture from the top and another from the side')
        self.send_msg.emit('Once you are ready, press the button above to save the picture\n')
        
        self.camera = cv2.VideoCapture(0)  
        while self.camera.isOpened():
            _, image = self.camera.read()
            if self.saveCurrentFrame:
                cv2.imwrite("currentFrame.jpg", image) # Save the current frame
                self.predictImage("currentFrame.jpg")
                self.saveCurrentFrame = 0
                
            self.displayImage(image)
            cv2.waitKey()
        
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
    
    
    def predictImage(self, image_path):
        prediction = GetPrediction(image_path)
        self.send_prediction.emit(prediction)
    
    @pyqtSlot()
    def savePicture(self):
        if self.camera:
            if self.camera.isOpened():
                self.saveCurrentFrame = 1
    
    
    @pyqtSlot()
    def closeCameraIfOpened(self):
        if self.camera:
            if self.camera.isOpened(): # Close the camera safely
                self.camera.release()
                cv2.destroyAllWindows()