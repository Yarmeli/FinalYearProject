# main.py
import sys, os, operator
import pandas as pd
from PyQt5 import QtCore, uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog

# Local files
import CameraWidget, ThumbWidget
from ThumbWidget import load_thumb_values
from helpers import Debug, DebugMode
from model import output_label, LoadSavedModel, GetPrediction
from image_segmentation import LoadSavedImageSegModel, SegmentSingleImage
from volume import CalculateVolume

# Find the correct path of the .ui file
ui_path = os.path.dirname(os.path.abspath(__file__))
form_class = uic.loadUiType(os.path.join(ui_path, "MainWindow.ui"))[0]

class MainWindow(QMainWindow, form_class):
    closeCamera = pyqtSignal() # Be able to communicate with the CameraWidget class
    def __init__(self):
        super().__init__()
        # Load the UI
        uic.loadUi("MainWindow.ui", self)
        
        
        LoadSavedModel() # Load the latest model for Image Classification
        LoadSavedImageSegModel() # Load the latest model for Image Segmentation
        
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
        self.camWidget.send_volumeInfo.connect(self.calculateVolume)
        
        # Signals for the Thumb Widget
        self.thumbWidget.send_thumbvalues.connect(self.setThumbValues)
        
    
    def initUI(self):
        # Set the initial message in the output box for the user
        self.setOutputText("Please choose 'Start Camera' or 'Upload Pictures'")
        self.setOutputText("You can change your thumb settings on the top left 'File' dropdown\n")
        if DebugMode:
            self.setOutputText("**** Debug Mode is ON, check the console for these messages ****")
    
        self.thumbValues = load_thumb_values()
        Debug("Thumb Values", f"Thumb values: {self.thumbValues}")
    
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
        
    @pyqtSlot(float, float, float)
    def setThumbValues(self, width, height, depth):
        self.thumbValues = [width, height, depth]
        self.setOutputText(f"Updated thumb values to: w: '{width}', h: '{height}' and d: '{depth}'")
    
    @pyqtSlot(dict, dict)
    def calculateVolume(self, predictions_dict, seg_images_path):
        try:
            food_items_for_volume = [x + 1 for x in [*predictions_dict]] # +1 because background = 0
            volume, segmented_predictions = CalculateVolume(seg_images_path, food_items_for_volume, self.thumbValues)
            Debug("Volume", f"Calculated volume: {volume}")
            
            self.estimateCalories(volume, predictions_dict, segmented_predictions)
        except Exception as e:
            self.setOutputText("**** The following issue occurred: {}".format(e)) 
            
        
        
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
            if any('top' in x.lower() or 'side' in x.lower() for x in fileNames): # Files are named correctly
                Debug("Upload", "Images have 'top' or 'side' in their name!")
                if 'top' in fileNames[0].lower() or 'side' in fileNames[1].lower():
                    latest_pics["top"] = fileNames[0]
                    latest_pics["side"] = fileNames[1]
                else:
                    latest_pics["top"] = fileNames[1]
                    latest_pics["side"] = fileNames[0]
            else: # File names are random
                Debug("Upload", "Images have random names!")
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
            
            Debug("Upload Class Prediction", f"ImageClassPred_dict values: {ImageClassPred_dict}")
                        
            # Send the messages to the user
            self.setOutputText(f"Loaded '{top_imgname}'")
            self.predictionMessage(prediction_top, confidence_top)
            self.setOutputText(f"Loaded '{side_imgname}'")
            self.predictionMessage(prediction_side, confidence_side)
            
            
            # Segment images
            latest_pics['top'] = SegmentSingleImage(latest_pics['top'])
            latest_pics['side'] = SegmentSingleImage(latest_pics['side'])
                        
            
            # Calculate Volume
            self.calculateVolume(ImageClassPred_dict, latest_pics)
    
    
    def closeEvent(self, event):
        self.closeCamera.emit() # Close camera if needed
        event.accept() # let the window close


    """     Calories Estimation Component      """
    
    def load_food_information(self, file = "Dataset/calories_density.csv"):
        Debug("Calorie Density Values", f"Loading information from '{file}'")
        information = pd.read_csv(file)
        full_dict = information.to_dict()
        food_calories = full_dict['Calories']
        food_density = full_dict['Density']
        return food_calories, food_density
    
    def estimateCalories(self, volume, image_class_predictions, image_seg_predictions):
        
        # Merge both of the predictions - Image Classification and Segmentation
        # Change the percentage of each class to be the average of both predictions
        merged_dict = { k: (image_class_predictions.get(k, 0) + image_seg_predictions.get(k, 0)) / 2 # Average
                      for k in set(image_class_predictions) | set(image_seg_predictions) }
        
        foodItem = max(merged_dict.items(), key=operator.itemgetter(1))[0] # Highest percentage
        
        Debug("Calories", f"Food Item Guessed: {output_label(foodItem)}")
        
        calories, density = self.load_food_information()
        
        mass = volume * density[foodItem]
        Debug("Calories", f"Estimated Mass: {mass}g")
        
        final_calories = (mass * calories[foodItem]) / 100
        final_calories = round(final_calories, 2)        
        self.setOutputText(f"Estimated Calories: {final_calories}kcal")
        
        return final_calories
        

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec_()
