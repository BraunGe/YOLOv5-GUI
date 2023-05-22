import sys                                                 #导入sys模块
from PyQt5.QtWidgets import QApplication, QMainWindow      #导入PyQt模块
from PyQt5 import QtCore, QtGui, QtWidgets

import main_win.mainUI as mainUI                       #导入刚刚UI文件生成的文件
from detect3 import CUi_MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)                           #使用sys新建一个应用（Application）对象
    timer_camera = QtCore.QTimer()  # 初始化定时器
    MainWindow = CUi_MainWindow()                             #新建一个Qt中QMainWindow()类函数
    #ui = mainUI.Ui_MainWindow()                            #定义ui，与我们设置窗体绑定
    #ui.setupUi(MainWindow)                                 #为MainWindow绑定窗体
    MainWindow.show()                                      #将MainWindow窗体进行显示
    sys.exit(app.exec_())                                  #进入主循环，事件开始处理，接收由窗口触发的事件



