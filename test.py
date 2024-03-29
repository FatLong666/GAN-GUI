from PyQt5 import Qt


def keyPressEvent(self, QKeyEvent):
    if QKeyEvent.key() == Qt.Key_Return:
         print('Space')
if __name__ == '__main__':
        keyPressEvent()
