import cv2, time
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage

from model import GetPrediction
from image_segmentation import SegmentSingleImage
from helpers import Debug

class CameraWidget(QWidget):
    send_msg = pyqtSignal(str)
    send_video = pyqtSignal(QImage)
    send_prediction = pyqtSignal(list, list)
    send_volumeInfo = pyqtSignal(dict, dict)
    
    def __init__(self):
        super(CameraWidget, self).__init__()
        self.camera = 0
        self.saveCurrentFrame = 0
        self.predictions_dict = {}
        
        
    @pyqtSlot()
    def startCapturing(self):
        if self.camera:
            self.send_msg.emit('WebCam already started!')
            return
        
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
                        Debug("Capture", "Resetting values (topPic, sidePic and predic_dict)")
                        topPictureTaken = 0
                        sidePictureTaken = 0
                        self.predictions_dict.clear()
                        
                    else:
                        # Take the side picture
                        side_pic_name = f"Pictures/Capture/side_picture_{timestr}.jpg"
                        cv2.imwrite(side_pic_name, image)
                        self.latest_pics["side"] = SegmentSingleImage(side_pic_name) # Keep track of segmented image
                        
                        self.send_msg.emit(f"Side picture taken and stored in '{side_pic_name}'")
                        self.send_msg.emit("You can retake the pictures by pressing the 'Take Picture' button\n")
                        self.predictImage(side_pic_name)
                        
                        sidePictureTaken = 1
                        
                        # Calculate the Volume
                        self.send_volumeInfo.emit(self.predictions_dict, self.latest_pics)
                        
                        
                if not topPictureTaken:
                    # Top picture has not been taken
                    # This is an 'if' instead of 'else' because topPictureTaken is set to 0 in line 42
                    # And setting this as 'if' allows this to be run again
                    top_pic_name = f"Pictures/Capture/top_picture_{timestr}.jpg"
                    cv2.imwrite(top_pic_name, image)
                    self.latest_pics["top"] = SegmentSingleImage(top_pic_name) # Keep track of segmented image
                    
                    self.send_msg.emit(f"Top picture taken and stored in '{top_pic_name}'")
                    self.send_msg.emit('Now take a side picture\n')
                    
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
        predictions, confidences = GetPrediction(image_path)
        temp_dict = dict(zip(predictions, confidences)) # Create a dict with these values
        """
            The code below stores the maximum value for each prediction
            max(dict1.get(), dict2.get()) returns the maximum value of key K between the dictionaries
            set(dict) returns a set of the dict keys and '.union(dict)' does set union operation
        """
        self.predictions_dict = {k : max(self.predictions_dict.get(k,0), temp_dict.get(k,0))
                                 for k in set(self.predictions_dict).union(set(temp_dict))}
        Debug("Classification Prediction", f"prediction_dict values: {self.predictions_dict}")
        self.send_prediction.emit(predictions, confidences)
    
    @pyqtSlot()
    def savePicture(self):
        if self.camera:
            if self.camera.isOpened():
                self.saveCurrentFrame = 1
        else:
            self.send_msg.emit('Please start the camera first')
    
    
    @pyqtSlot()
    def closeCameraIfOpened(self):
        if self.camera:
            if self.camera.isOpened(): # Close the camera safely
                self.camera.release()
                cv2.destroyAllWindows()