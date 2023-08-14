import os
from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.gui import (QgsFieldComboBox, QgsMapLayerComboBox)
from qgis.utils import iface


FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'howole_dialog_base.ui'))

class HowoleDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super(HowoleDialog, self).__init__(parent)
        self.setupUi(self)
        # создаем подключение на событие нажатия кнопки (pushButton_analiz), создаем имя функции события "btnAnaliztClicked"
       # self.pushButton_analiz.clicked.connect(self.btnAnaliztClicked)
                
   # def btnAnaliztClicked(self):
       # layer = iface.activeLayer()
        #print(layer)
    
    



