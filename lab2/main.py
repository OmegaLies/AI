import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from transformers import GPT2LMHeadModel, GPT2Tokenizer

np.random.seed(42)
torch.manual_seed(42)

def load_tokenizer_and_model(model_name_or_path):
  return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()

def generate(
    model, tok, text,
    do_sample=True, max_length=50, repetition_penalty=5.0,
    top_k=5, top_p=0.95, temperature=1,
    num_beams=None,
    no_repeat_ngram_size=3
    ):
  input_ids = tok.encode(text, return_tensors="pt").cuda()
  out = model.generate(
      input_ids.cuda(),
      max_length=max_length,
      repetition_penalty=repetition_penalty,
      do_sample=do_sample,
      top_k=top_k, top_p=top_p, temperature=temperature,
      num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
      )
  return list(map(tok.decode, out))

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(583, 273)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(80, 170, 131, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.OnClick)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(40, 140, 91, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(40, 40, 211, 61))
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(120, 10, 51, 16))
        self.label.setObjectName("label")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(290, 40, 251, 171))
        self.textBrowser.setObjectName("textBrowser")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 110, 91, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(350, 10, 131, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(170, 110, 71, 21))
        self.label_4.setObjectName("label_4")
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(160, 140, 91, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ruGPT3 models"))
        self.pushButton.setText(_translate("MainWindow", "Сгенерировать текст"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Small"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Medium"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Large"))
        self.label.setText(_translate("MainWindow", "Затравка"))
        self.label_2.setText(_translate("MainWindow", "Выберите модель"))
        self.label_3.setText(_translate("MainWindow", "Сгенерированный текст"))
        self.label_4.setText(_translate("MainWindow", "Кол-во слов"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "50"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "75"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "100"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "125"))
        self.comboBox_2.setItemText(4, _translate("MainWindow", "150"))
        self.comboBox_2.setItemText(5, _translate("MainWindow", "200"))

    def OnClick(self):
        input_text = self.textEdit.toPlainText()
        model = self.comboBox.currentText()
        words = int(self.comboBox_2.currentText())
        if input_text != "":
            if model == "Small":
                tok_small, model_small = load_tokenizer_and_model("sberbank-ai/rugpt3small_based_on_gpt2")
                generated = generate(model_small, tok_small, input_text, num_beams=10, max_length=words)
            elif model == "Medium":
                tok_medium, model_medium = load_tokenizer_and_model("sberbank-ai/rugpt3medium_based_on_gpt2")
                generated = generate(model_medium, tok_medium, input_text, num_beams=10, max_length=words)
            else:
                tok_large, model_large = load_tokenizer_and_model("sberbank-ai/rugpt3large_based_on_gpt2")
                generated = generate(model_large, tok_large, input_text, num_beams=10, max_length=words)
            self.textBrowser.setText(generated[0])
        else:
            self.textBrowser.setText("Отсутствует затравка, генерация невозможна")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())