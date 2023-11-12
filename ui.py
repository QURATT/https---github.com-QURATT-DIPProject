import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QFileDialog

class MenuPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        label = QLabel("Choose an option:", self)
        layout.addWidget(label)

        button_texture_synthesis = QPushButton("Texture Synthesis", self)
        button_texture_synthesis.clicked.connect(self.on_texture_synthesis_clicked)
        layout.addWidget(button_texture_synthesis)

        button_image_quilting = QPushButton("Image Quilting", self)
        button_image_quilting.clicked.connect(self.on_image_quilting_clicked)
        layout.addWidget(button_image_quilting)

        button_texture_transfer = QPushButton("Texture Transfer", self)
        button_texture_transfer.clicked.connect(self.on_texture_transfer_clicked)
        layout.addWidget(button_texture_transfer)

    def on_texture_synthesis_clicked(self):
        self.parent().on_option_selected("Texture Synthesis")

    def on_image_quilting_clicked(self):
        self.parent().on_option_selected("Image Quilting")

    def on_texture_transfer_clicked(self):
        self.parent().on_option_selected("Texture Transfer")


class ImageSelectionPage(QWidget):
    def __init__(self, option, parent=None):
        super().__init__(parent)
        self.option = option
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        label = QLabel(f"Choose an image for {self.option}:", self)
        layout.addWidget(label)

        button_select_image = QPushButton("Select Image", self)
        button_select_image.clicked.connect(self.open_file_dialog)
        layout.addWidget(button_select_image)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Select an image", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        if filename:
            self.parent().on_image_selected(filename)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Project GUI")
        self.menu_page = MenuPage(self)
        self.image_selection_page = None
        self.setup_ui()

    def setup_ui(self):
        self.setCentralWidget(self.menu_page)

    def on_option_selected(self, option):
        self.image_selection_page = ImageSelectionPage(option, self)
        self.setCentralWidget(self.image_selection_page)

    def on_image_selected(self, filename):
        print("Selected file:", filename)
        # Placeholder method for processing the selected image
        # Replace with your actual code for image processing


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
