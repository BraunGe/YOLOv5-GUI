from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.Qt import *
from main_win.mainUI import Ui_MainWindow
import cv2
 
class CUi_MainWindow(QMainWindow, Ui_MainWindow): #继承于UI父类
    returnSignal = pyqtSignal()

    def __init__(self, parent=None):
        super(CUi_MainWindow, self).__init__(parent)
        self.timer_camera = QTimer() #初始化定时器
        self.cap = cv2.VideoCapture() #初始化摄像头
        self.CAM_NUM = 0
        self.setupUi(self)
        self.slot_init()
        
    def slot_init(self):
        self.timer_camera.timeout.connect(self.show_camera)
        self.pushButton_2.clicked.connect(self.openCamera)
        self.pushButton.clicked.connect(self.closeCamera)
    #    self.pushButton.clicked.connect(self.btn)  

    def show_camera(self):
        flag,self.image = self.cap.read()
        show = cv2.resize(self.image,(480,320))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1],show.shape[0],QImage.Format_RGB888)
        self.videolabel.setPixmap(QPixmap.fromImage(showImage))

 #打开关闭摄像头控制
    def slotCameraButton(self):
         if self.timer_camera.isActive() == False:
            self.openCamera()
         else:
            self.closeCamera()

 #打开摄像头
    def openCamera(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
            buttons=QMessageBox.Ok,
            defaultButton=QMessageBox.Ok)
        else:
            self.timer_camera.start(30)

#关闭摄像头
    def closeCamera(self):
        self.timer_camera.stop()
        self.cap.release()
        self.videolabel.clear()



