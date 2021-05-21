from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap, QDoubleValidator
from PyQt5.QtWidgets import QWidget

class ThumbWidget(QWidget):
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
        
        
        
        """
        Pending work:
            Send values to the main application
        """