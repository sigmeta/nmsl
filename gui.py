# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import sys
import json
import time
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from functools import partial
from main import str2emoji,str2emoji_deep,emoji2str
from models.converter import TextConverter
from generate import MAX_VOCAB


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(523, 387)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(160, 70, 121, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(290, 70, 121, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(160, 110, 121, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(290, 110, 121, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(160, 40, 251, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(50, 200, 361, 141))
        self.textBrowser.setObjectName("textBrowser")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 40, 121, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 170, 361, 21))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "抽象话转换机"))
        self.pushButton.setText(_translate("MainWindow", "普通转抽象"))
        self.pushButton_2.setText(_translate("MainWindow", "普通转深度抽象"))
        self.pushButton_3.setText(_translate("MainWindow", "抽象转普通"))
        self.pushButton_4.setText(_translate("MainWindow", "深度抽象转普通"))
        self.label.setText(_translate("MainWindow", "在此输入文本："))
        self.label_2.setText(_translate("MainWindow", "转换结果："))


def s2e(ui,dct, dct_pinyin):
    print("s2e")
    text=ui.lineEdit.text()
    res=str2emoji(text,dct,dct_pinyin)
    print(res)
    ui.textBrowser.setText(res)

def s2edeep(ui,dct,dct_pinyin):
    print("s2e deep")
    text=ui.lineEdit.text()
    res=str2emoji_deep(text,dct,dct_pinyin)
    print(res)
    ui.textBrowser.setText(res)


def e2s(ui,model,convert):
    text = ui.lineEdit.text()
    src = torch.tensor(convert.text_to_arr(text))
    src = src.unsqueeze(0)
    stime=time.time()
    tgt_list = [1]
    for i in range(64):
        tgt = torch.tensor(tgt_list)
        tgt = tgt.unsqueeze(0)
        out = model(src, tgt)
        if int(out.argmax(-1)[-1, 0]) == 1:
            break
        tgt_list.append(int(out.argmax(-1)[-1, 0]))
    res=convert.arr_to_text(tgt_list[1:])
    etime=time.time()
    print(res,f"Translation cost {etime-stime} seconds")
    ui.textBrowser.setText(res)


if __name__ == '__main__':
    print("正在加载模型，别急")
    stime=time.time()
    dct = json.loads(open('data/dict.json', encoding='utf8').read())
    dct_pinyin = json.loads(open('data/dict_pinyin.json', encoding='utf8').read())
    model1 = torch.load("output/e2s.pkl")
    model1.eval()
    model2 = torch.load("output/e2sdeep.pkl")
    model2.eval()
    convert1 = TextConverter("data/train/e2s/vocab.txt")
    convert2 = TextConverter("data/train/e2sdeep/vocab.txt")
    etime=time.time()
    print(f"加载模型耗时{etime-stime}秒")
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.pushButton.clicked.connect(partial(s2e, ui,dct,dct_pinyin))
    ui.pushButton_2.clicked.connect(partial(s2edeep, ui, dct, dct_pinyin))
    ui.pushButton_3.clicked.connect(partial(e2s, ui, model1, convert1))
    ui.pushButton_4.clicked.connect(partial(e2s, ui, model2, convert2))
    sys.exit(app.exec_())