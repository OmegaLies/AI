import base64
import json
import requests
from pathlib import Path
from PIL import Image, ImageDraw
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


def img_name(img_path):
    return Path(img_path).name


def image_distortion(img_path, quality):
    img = Image.open(f'{img_path}')
    img.save(f'output/{img_name(img_path)}', quality=quality)


def select_area(vertices, img_path):
    img = Image.open(f'{img_path}')
    draw = ImageDraw.Draw(img)
    for vertical in vertices:
        draw.rectangle(vertical, outline='blue', width=10)
    img.save(f'output/{img_name(img_path)}')
    return img


# General function with POST requets to service
def detect_faces(img_path, quality):
    image_distortion(img_path, quality)
    image_file = open(f'output/{img_name(img_path)}', 'rb')
    image_b64 = base64.b64encode(image_file.read())
    with open("yandex_auth_token.txt", 'r', encoding='utf-8') as key_f:
        api_key = key_f.read()

    request_data_dict = {"folderId": "b1gucq1td3r5l607qgnu",
                         "analyze_specs": [
                             {"content": image_b64.decode(),
                              "features": [{"type": "FACE_DETECTION"}],
                              "mimeType": "image/jpeg"}]}

    request_header = {"Authorization": f"Api-Key {api_key}"}
    response = requests.post("https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze",
                             json=request_data_dict,
                             headers=request_header)
    response_data = response.json()
    # print(response)
    # print(response_data)
    with open('output.json', 'w') as res_f:
        json.dump(response.json(), res_f)
    if response_data['results'][0]['results'][0]['faceDetection'] == {}:
        result_image = select_area([], img_path)
        return [result_image, 0]
    response_data = response_data['results'][0]['results'][0]['faceDetection']['faces']
    faces_count = len(response_data)

    detected_faces_vertices = list()
    for face in response_data:
        for vertical in face['boundingBox']['vertices'][::2]:
            detected_faces_vertices.append((int(vertical['x']), int(vertical['y'])))

    chunked = list()
    for i in range(0, len(detected_faces_vertices), 2):
        chunked.append(detected_faces_vertices[i:i + 2])
    print('Coordinates of faces: ', chunked)
    detected_faces_vertices = chunked
    result_image = select_area(detected_faces_vertices, img_path)

    return [result_image, faces_count]


class Ui_MainWindow(object):
    filename = ""

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1049, 391)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(30, 220, 141, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.detection)
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setGeometry(QtCore.QRect(120, 190, 42, 22))
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(100)
        self.spinBox.setObjectName("spinBox")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(330, 30, 131, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(760, 30, 131, 21))
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 150, 141, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.select_image)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(40, 190, 61, 21))
        self.label_3.setObjectName("label_3")
        self.label_Image_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_Image_1.setGeometry(QtCore.QRect(200, 70, 401, 291))
        self.label_Image_1.setObjectName("label_Image_1")
        self.label_Image_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_Image_2.setGeometry(QtCore.QRect(630, 70, 401, 291))
        self.label_Image_2.setObjectName("label_Image_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Обнаружение лиц"))
        self.pushButton.setText(_translate("MainWindow", "Провести обнаружение"))
        self.label.setText(_translate("MainWindow", "Выбранное изображение"))
        self.label_2.setText(_translate("MainWindow", "Результат обнаружения"))
        self.pushButton_2.setText(_translate("MainWindow", "Выбрать изображение"))
        self.label_3.setText(_translate("MainWindow", "Качество"))

    def select_image(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName(
            None, "Выбрать файл", ".", "JPG File(*.jpg);;JPEG Files(*.jpeg);;PNG Files(*.png);;All Files(*)")[0]
        image_profile = QtGui.QImage(self.filename)  # QImage object
        image_profile = image_profile.scaled(300, 300, aspectRatioMode=Qt.KeepAspectRatio,
                                             transformMode=Qt.SmoothTransformation)
        self.label_Image_1.setPixmap(QtGui.QPixmap.fromImage(image_profile))

    def detection(self):
        quality = self.spinBox.value()
        print(f"Quality: {quality}")
        detect_faces(self.filename, quality)
        image_profile = QtGui.QImage(rf'output/{img_name(self.filename)}')  # QImage object
        image_profile = image_profile.scaled(300, 300, aspectRatioMode=Qt.KeepAspectRatio,
                                             transformMode=Qt.SmoothTransformation)
        self.label_Image_2.setPixmap(QtGui.QPixmap.fromImage(image_profile))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
