# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\JingZ\source\repos\UI\UI\main_win\UI4.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1454, 832)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(11, 111, 201, 41))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2.setCheckable(True)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(220, 61, 1221, 721))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.videolabel = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.videolabel.setMinimumSize(QtCore.QSize(1280, 720))
        self.videolabel.setObjectName("videolabel")
        self.verticalLayout_2.addWidget(self.videolabel)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(11, 61, 201, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_3.setCheckable(True)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_3.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_4.setCheckable(False)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_3.addWidget(self.pushButton_4)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(10, 390, 201, 391))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.label.setObjectName("label")
        self.verticalLayout_4.addWidget(self.label)
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(11, 161, 201, 81))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.lineEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget_5)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout_5.addWidget(self.lineEdit)
        self.pushButton_6 = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.pushButton_6.setCheckable(True)
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout_5.addWidget(self.pushButton_6)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, 201, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.radioButton_6 = QtWidgets.QRadioButton(self.horizontalLayoutWidget_2)
        self.radioButton_6.setObjectName("radioButton_6")
        self.horizontalLayout.addWidget(self.radioButton_6)
        self.radioButton_5 = QtWidgets.QRadioButton(self.horizontalLayoutWidget_2)
        self.radioButton_5.setObjectName("radioButton_5")
        self.horizontalLayout.addWidget(self.radioButton_5)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(220, 10, 738, 41))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.radioButton_2 = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.radioButton_2.setObjectName("radioButton_2")
        self.horizontalLayout_2.addWidget(self.radioButton_2)
        self.radioButton = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.radioButton.setObjectName("radioButton")
        self.horizontalLayout_2.addWidget(self.radioButton)
        self.radioButton_4 = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.radioButton_4.setObjectName("radioButton_4")
        self.horizontalLayout_2.addWidget(self.radioButton_4)
        self.radioButton_3 = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.radioButton_3.setObjectName("radioButton_3")
        self.horizontalLayout_2.addWidget(self.radioButton_3)
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.radioButton_7 = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.radioButton_7.setObjectName("radioButton_7")
        self.horizontalLayout_2.addWidget(self.radioButton_7)
        self.radioButton_9 = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.radioButton_9.setObjectName("radioButton_9")
        self.horizontalLayout_2.addWidget(self.radioButton_9)
        self.radioButton_8 = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.radioButton_8.setObjectName("radioButton_8")
        self.horizontalLayout_2.addWidget(self.radioButton_8)
        self.radioButton_10 = QtWidgets.QRadioButton(self.horizontalLayoutWidget_3)
        self.radioButton_10.setObjectName("radioButton_10")
        self.horizontalLayout_2.addWidget(self.radioButton_10)
        self.verticalLayoutWidget_6 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_6.setGeometry(QtCore.QRect(10, 250, 201, 61))
        self.verticalLayoutWidget_6.setObjectName("verticalLayoutWidget_6")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_6)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.comboBox = QtWidgets.QComboBox(self.verticalLayoutWidget_6)
        self.comboBox.setObjectName("comboBox")
        self.verticalLayout_6.addWidget(self.comboBox)
        self.pushButton_7 = QtWidgets.QPushButton(self.verticalLayoutWidget_6)
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout_6.addWidget(self.pushButton_7)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(10, 320, 201, 61))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_3.addWidget(self.pushButton)
        self.pushButton_5 = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout_3.addWidget(self.pushButton_5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1454, 23))
        self.menubar.setObjectName("menubar")
        self.menuCamera = QtWidgets.QMenu(self.menubar)
        self.menuCamera.setObjectName("menuCamera")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuCamera.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_2.setText(_translate("MainWindow", "Camera Run/Pause"))
        self.videolabel.setText(_translate("MainWindow", "Video_Window"))
        self.pushButton_3.setText(_translate("MainWindow", "Picture Run"))
        self.pushButton_4.setText(_translate("MainWindow", "Next"))
        self.label.setText(_translate("MainWindow", "Label_Window"))
        self.pushButton_6.setText(_translate("MainWindow", "Start RTMP"))
        self.radioButton_6.setText(_translate("MainWindow", "640*640"))
        self.radioButton_5.setText(_translate("MainWindow", "1280*1280"))
        self.label_2.setText(_translate("MainWindow", "FOR GPU"))
        self.radioButton_2.setText(_translate("MainWindow", "Small Base"))
        self.radioButton.setText(_translate("MainWindow", "Small SOD"))
        self.radioButton_4.setText(_translate("MainWindow", "Small P6"))
        self.radioButton_3.setText(_translate("MainWindow", "Nano P6"))
        self.label_3.setText(_translate("MainWindow", "|FOR CPU"))
        self.radioButton_7.setText(_translate("MainWindow", "Small Base"))
        self.radioButton_9.setText(_translate("MainWindow", "Small P6"))
        self.radioButton_8.setText(_translate("MainWindow", "Small SOD"))
        self.radioButton_10.setText(_translate("MainWindow", "Nano P6"))
        self.pushButton_7.setText(_translate("MainWindow", "Run/Pause"))
        self.pushButton.setText(_translate("MainWindow", "Restart"))
        self.pushButton_5.setText(_translate("MainWindow", "Quit"))
        self.menuCamera.setTitle(_translate("MainWindow", "Camera"))