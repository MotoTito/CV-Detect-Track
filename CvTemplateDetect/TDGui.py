import glob
import sys
from PyQt5 import QtGui
from PyQt5 import QtWidgets
import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QGridLayout, QSlider, QSpinBox, QVBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget
import cv2 as cv
from matplotlib import pyplot as plt
from CvTemplateDetect import SelectColorRange
import numpy as np

class ImageWindow(QtWidgets.QWidget):
  image = None
  imageList = []
  def __init__(self, image=None, imageList = []):
      QWidget.__init__(self)
      self.label = QLabel(self)
      self.label.resize(800, 600)
      self.image = image
      self.imageList = imageList
      if len(image) > 0:
        convertedImage = QtGui.QImage(image, image.shape[1],image.shape[0], QtGui.QImage.Format_RGB888)
        pixmap1 = QtGui.QPixmap.fromImage(convertedImage)
        self.pixmap = pixmap1.scaled(self.width(), self.height(),Qt.AspectRatioMode.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)
        self.label.setMinimumSize(1, 1)
      else:
        self.setGeometry(100,100,800,600)

  def updateImageMask(self, lowerRGB, upperRGB):
    mask = SelectColorRange(self.image, lowerRGB, upperRGB)
    tempImg = cv.bitwise_and(self.image, self.image, mask=mask)
    convertedImage = QtGui.QImage(tempImg, mask.shape[1],mask.shape[0], QtGui.QImage.Format_RGB888)
    # convertedImage = QtGui.QImage(mask, mask.shape[1],mask.shape[0], QtGui.QImage.Format_Grayscale8).rgbSwapped()
    # convertedImage = QtGui.QImage(mask, self.width(),self.height(), QtGui.QImage.Format_Mono)
    pixmap1 = QtGui.QPixmap.fromImage(convertedImage)
    tempPixmap = pixmap1.scaled(self.width(), self.height(),Qt.AspectRatioMode.KeepAspectRatio)
    self.label.setPixmap(tempPixmap)
    self.label.resize(self.width(), self.height())
    self.label.setMinimumSize(1, 1)

  def resizeEvent(self, event):
    if hasattr(self.label, "pixmap") and self.label.pixmap != None:
      convertedImage = QtGui.QImage(self.image, self.image.shape[1],self.image.shape[0], QtGui.QImage.Format_RGB888)
      pixmap1 = QtGui.QPixmap.fromImage(convertedImage)
      self.pixmap = pixmap1.scaled(self.width(), self.height(),Qt.AspectRatioMode.KeepAspectRatio)
      self.label.setPixmap(self.pixmap)
      self.label.resize(self.width(), self.height())

class ToolWindow(QtWidgets.QWidget):

  lowerRGB = np.asarray([0,0,0])
  upperRGB = np.asarray([255,255,255])

  def __init__(self):
    QWidget.__init__(self)
    self.setWindowTitle('Tool Window')
    self.setGeometry(100,100,250,500)
    grid = QGridLayout()
    grid.addWidget(self.ColorSliders("Red_Channel"))
    grid.addWidget(self.ColorSliders("Green_Channel"))
    grid.addWidget(self.ColorSliders("Blue_Channel"))
    self.setLayout(grid)

  def ColorSliders(self, labelName):
    groupBox = QtWidgets.QGroupBox(labelName)
    lowerSlider = QSlider(Qt.Horizontal, parent=groupBox)
    lowerSlider.setObjectName("lower_"+labelName)
    lowerSlider.setRange(0,255)
    lowerValue = QSpinBox()
    lowerValue.setRange(0,255)

    upperSlider = QSlider(Qt.Horizontal, parent=groupBox)
    upperSlider.setRange(0,255)
    upperValue = QSpinBox()
    upperValue.setRange(0,255)
    upperSlider.setObjectName("upper_"+labelName)

    lowerSlider.valueChanged.connect(lowerValue.setValue)
    upperSlider.valueChanged.connect(upperValue.setValue)
    lowerValue.valueChanged.connect(lowerSlider.setValue)
    upperValue.valueChanged.connect(upperSlider.setValue)
    lowerValue.valueChanged.connect(self.UpdateRGB)
    upperValue.valueChanged.connect(self.UpdateRGB)

    upperSlider.setValue(255)

    vbox = QVBoxLayout()
    vbox.addWidget(lowerSlider)
    vbox.addWidget(lowerValue)
    vbox.addWidget(upperSlider)
    vbox.addWidget(upperValue)
    vbox.addStretch(1)
    groupBox.setLayout(vbox)
    return groupBox

  def UpdateRGB(self):
    redLower = self.findChild(QSlider,"lower_Red_Channel")
    greenLower = self.findChild(QSlider,"lower_Green_Channel")
    blueLower = self.findChild(QSlider,"lower_Blue_Channel")
    redupper = self.findChild(QSlider,"upper_Red_Channel")
    greenupper = self.findChild(QSlider,"upper_Green_Channel")
    blueupper = self.findChild(QSlider,"upper_Blue_Channel")
    if redLower == None:
      return
    self.lowerRGB = np.asarray([redLower.value(), greenLower.value(), blueLower.value()])
    self.upperRGB = np.asarray([redupper.value(),greenupper.value(), blueupper.value()])
    ImageWindow.updateImageMask(imageWindow, self.lowerRGB,self.upperRGB)

app = QApplication(sys.argv)
testImg = plt.imread("./TestImages/sc4.jpg")
testImages = glob.glob("./TestImages/*.jpg")
testImgArray = []

for imgPath in testImages:
    testImgArray.append(plt.imread(imgPath))

convertedImage = QtGui.QImage(testImg, testImg.shape[1],testImg.shape[0], QtGui.QImage.Format_RGB888)
imageWindow = ImageWindow(testImg, testImgArray)
imageWindow.setWindowTitle('Image Display')

helloMsg = QLabel('<h1>Hello World</h1>', parent=imageWindow)

toolWindow = ToolWindow()

imageWindow.label.setPixmap(QtGui.QPixmap.fromImage(convertedImage))
imageWindow.show()
toolWindow.show()
sys.exit(app.exec_())