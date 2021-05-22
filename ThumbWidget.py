import pandas as pd
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap, QDoubleValidator
from PyQt5.QtWidgets import QWidget, QErrorMessage
from PyQt5.QtCore  import pyqtSignal

from helpers import Debug

class ThumbWidget(QWidget):
    send_thumbvalues = pyqtSignal(float, float, float)
    def __init__(self):
        super().__init__()
        uic.loadUi("thumb.ui", self)
        
        self.initUI()

        self.saveButton.clicked.connect(self.sendValues)
        self.cancelButton.clicked.connect(self.close)
                
    def initUI(self):
        topview_path = "Pictures/Thumb_info_top.png"
        sideview_path = "Pictures/Thumb_info_side.png"
        
        topimage = QImage(topview_path)
        sideimage = QImage(sideview_path)
        
        self.infoTop.setPixmap(QPixmap.fromImage(topimage))
        self.infoSide.setPixmap(QPixmap.fromImage(sideimage))
        
        
        floatOnly = QDoubleValidator(0.0, 5.0, 2)
        self.widthLineEdit.setValidator(floatOnly)
        self.heightLineEdit.setValidator(floatOnly)
        self.depthLineEdit.setValidator(floatOnly)
        
    def sendValues(self):
        try:
            width = float(self.widthLineEdit.text())
            height = float(self.heightLineEdit.text())
            depth = float(self.depthLineEdit.text())
            
            self.send_thumbvalues.emit(width, height, depth)
            save_thumb_values([[width, height, depth]])
            self.close()
        except:
            error_dialog = QErrorMessage(self)
            error_dialog.showMessage("Please add valid values!")
        
def load_thumb_values(file = "thumbValues.csv"):
    df = pd.read_csv(file)
    Debug("Thumb Values", f"Loading thumb values from '{file}'")
    return df.values[1]

def save_thumb_values(values, file = "thumbValues.csv"):
    Debug("Thumb Values", f"Saving thumb values to '{file}'")
    with open(file, 'w') as csv_file:
        csv_file.write("# User's thumb values in cm\n")
        header = ['width', 'height', 'depth']
        pd.DataFrame(values).to_csv(csv_file, header=header)
            
