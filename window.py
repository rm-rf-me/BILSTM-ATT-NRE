# -*- coding: utf-8 -*-

'''
@Time    : 2020/12/20 下午6:04
@Author  : liou
@FileName: window.py
@Software: PyCharm
 
'''
from test import *  #包含模块
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import numpy as np
from train import trainer



class Function(QWidget, Ui_MainWindow):
    # 构造函数
    def __init__(self):
        # 继承图形界面的所有成员
        super(Function, self).__init__()
        # 建立并显示界面
        self.setupUi(self)
        self.show()

        self.pushButton.clicked.connect(self.Determination)

    def Determination(self):
        seq = self.lineEdit.text()
        peo1 = self.lineEdit_2.text()
        peo2 = self.lineEdit_3.text()

        self.haha = trainer()
        ans = self.haha.test(seq, peo1, peo2)

        self.lineEdit_4.setText(ans)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    fu = Function() #创建对象，会调用构造函数
    sys.exit(app.exec_())