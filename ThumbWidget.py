from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget

class ThumbWidget(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("thumb.ui", self)
        
        topview_path = "./Thumb_info_top.png"
        sideview_path = "./Thumb_info_side.png"
        
        topimage = QImage(topview_path)
        sideimage = QImage(sideview_path)
        
        self.infoTop.setPixmap(QPixmap.fromImage(topimage))
        self.infoSide.setPixmap(QPixmap.fromImage(sideimage))