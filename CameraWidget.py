import cv2, time
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage

from model import GetPrediction
from helpers import Debug

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
        
        timestr, top_pic_name, side_pic_name = "", "", ""        
        self.latest_pics = {'top': top_pic_name, 'side': side_pic_name}
        topPictureTaken, sidePictureTaken = 0, 0
        
        while self.camera.isOpened():
            _, image = self.camera.read()
            if self.saveCurrentFrame:
                
                timestr = time.strftime('%Y%m%d-%H%M%S') # Get current time
                
                if topPictureTaken: 
                # Top picture has been taken, now check the side picture
                
                    if sidePictureTaken:
                        # Both pictures have been taken, retake both pictures
                        self.send_msg.emit('\nRetaking both pictures!')
                        topPictureTaken = 0
                        sidePictureTaken = 0
                        
                    else:
                        # Take the side picture
                        side_pic_name = f"Pictures/side_picture_{timestr}.jpg"
                        cv2.imwrite(side_pic_name, image)
                        Debug("Capture", f"Side picture taken and stored in {side_pic_name}")
                        self.predictImage(side_pic_name)
                        
                        sidePictureTaken = 1
                        
                if not topPictureTaken:
                    # Top picture has not been taken
                    # This is an 'if' instead of 'else' because topPictureTaken is set to 0 in line 42
                    # And setting this as 'if' allows this to be run again
                    top_pic_name = f"Pictures/top_picture_{timestr}.jpg"
                    cv2.imwrite(top_pic_name, image)
                    Debug("Capture", f"Top picture taken and stored in {top_pic_name}")
                    
                    self.predictImage(top_pic_name)
                    topPictureTaken = 1
                    
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