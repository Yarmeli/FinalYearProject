# main.py
import sys, os
from PyQt5 import QtCore, uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog

# Local files
import CameraWidget, ThumbWidget
from helpers import Debug, DebugMode
from model import output_label, LoadSavedModel, GetPrediction

# Find the correct path of the .ui file
ui_path = os.path.dirname(os.path.abspath(__file__))
form_class = uic.loadUiType(os.path.join(ui_path, "MainWindow.ui"))[0]

class MainWindow(QMainWindow, form_class):
    closeCamera = pyqtSignal() # Be able to communicate with the CameraWidget class
    def __init__(self):
        super().__init__()
        # Load the UI
        uic.loadUi("MainWindow.ui", self)
        
        # Load the latest model for Image Classification
        LoadSavedModel()
        
        self.initUI()
        
        # Create an object for each of the other classes
        self.camWidget = CameraWidget.CameraWidget()
        self.thumbWidget = ThumbWidget.ThumbWidget()
        
                    # Setup the Signals and the Slots
        
        # Signals for the User Interface
        self.startCamera.clicked.connect(self.camWidget.startCapturing)
        self.takePicture.clicked.connect(self.camWidget.savePicture)
        self.uploadPicture.clicked.connect(self.uploadFiles)
        
        # Actions Dropdown menu - Under the 'File' in the top left
        self.actionThumb_Settings.triggered.connect(self.thumbWidget.show)
        self.actionClose.triggered.connect(self.close)
        
        # Close the camera correctly
        self.closeCamera.connect(self.camWidget.closeCameraIfOpened)
        
        # Signals for the Camera Widget class
        self.camWidget.send_msg.connect(self.setOutputText)
        self.camWidget.send_video.connect(self.setVideoFeed)
        self.camWidget.send_prediction.connect(self.predictionMessage)
        
        
    
    def initUI(self):
        # Set the initial message in the output box for the user
        self.setOutputText("Please choose 'Start Camera' or 'Upload Pictures'")
        self.setOutputText("You can change your thumb settings on the top left 'File' dropdown\n")
        if DebugMode:
            self.setOutputText("**** Debug Mode is ON, check the console for these messages ****")
    
    
    """     SLOTS    """
    
        
    @pyqtSlot(str)
    def setOutputText(self, text):
        Debug("Output", text)
        self.output_box.append(text)    
    
    @pyqtSlot(QImage)
    def setVideoFeed(self, img):
        self.videoFeed.setPixmap(QPixmap.fromImage(img))
        self.videoFeed.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    
    @pyqtSlot(list, list)
    def predictionMessage(self, predictions, confidences):
        message = "Model predictions:\n"
        for i in range(len(predictions)):
            message += "{}: {:.1f}%\n".format(output_label(predictions[i]), confidences[i])
        self.setOutputText(message)
    
    """     METHODS    """
    
    
    def uploadFiles(self):
        fileNames, _ = QFileDialog.getOpenFileNames(self,
            "Open Images", "", "Image Files (*.png *.jpg *.bmp)");
        if fileNames:
            if len(fileNames) != 2:
                self.setOutputText("Please upload exactly 2 images! (the top and the side view)")
                return

            """
            Selecting which of the two images will be considered the 'top' and 'side' view
            Check image names - Does any of them have 'top' or 'side'?
            If none of them have it, first image is the top view and the second the side view
            """               
            latest_pics = {}
            if any('top' in x or 'side' in x for x in fileNames): # Files are named correctly
                if 'top' in fileNames[0] or 'side' in fileNames[1]:
                    latest_pics["top"] = fileNames[0]
                    latest_pics["side"] = fileNames[1]
                elif 'top' in fileNames[1] or 'side' in fileNames[0]:
                    latest_pics["top"] = fileNames[1]
                    latest_pics["side"] = fileNames[0]
            else: # File names are random
                 latest_pics["top"] = fileNames[0] # first image is top view
                 latest_pics["side"] = fileNames[1] # second image is side view
                    
            top_imgname = latest_pics['top'][latest_pics['top'].rindex('/') + 1:]
            side_imgname = latest_pics['side'][latest_pics['side'].rindex('/') + 1:]
            
            Debug("Upload", f"Top image: '{top_imgname}', Side image: '{side_imgname}'")
            
            prediction_top, confidence_top = GetPrediction(latest_pics["top"])
            prediction_side, confidence_side = GetPrediction(latest_pics["side"])
            
            top_pred_dict = dict(zip(prediction_top, confidence_top))
            side_pred_dict = dict(zip(prediction_side, confidence_side))
            
            ImageClassPred_dict = {k : max(top_pred_dict.get(k,0), side_pred_dict.get(k,0))
                                 for k in set(top_pred_dict).union(set(side_pred_dict))}
            
            Debug("Classification Prediction", f"ImageClassPred_dict values: {ImageClassPred_dict}")
                        
            # Send the messages to the user
            self.setOutputText(f"Loaded '{top_imgname}'")
            self.predictionMessage(prediction_top, confidence_top)
            self.setOutputText(f"Loaded '{side_imgname}'")
            self.predictionMessage(prediction_side, confidence_side)
    
    
    def closeEvent(self, event):
        self.closeCamera.emit() # Close camera if needed
        event.accept() # let the window close
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec_()
