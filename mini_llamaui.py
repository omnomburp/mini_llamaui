import sys
import io
import ollama
import mistune
import math
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QSizePolicy, QScrollArea, QAbstractScrollArea

class MarkdownTextEdit(QTextEdit):
    def __init__(self):
        super().__init__()

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

    def setMarkdown(self, markdown_text):
        html_text = mistune.markdown(markdown_text)
        self.setHtml(html_text)
        QTimer.singleShot(0, self.adjustHeight)

    def adjustHeight(self):
        # Get the document size and set the height accordingly
        new_height = self.document().size().height() + 10  # Add some padding
        self.setMinimumHeight(math.ceil(new_height))
        self.setMaximumHeight(math.ceil(new_height))

class LlamaWorker(QThread):
    llama_finished = pyqtSignal(str, MarkdownTextEdit)
    llama_ended = pyqtSignal(str)

    def __init__(self, conversation, chat_box):
        super().__init__()
        self.conservation = conversation
        self.chat_box = chat_box

    def run(self):
        stream = ollama.chat(
            model='openhermes:7b-mistral-v2.5-q6_K',
            messages=self.conservation,
            stream=True,
        )

        result_text = ""
        for chunk in stream:
            result_text += chunk["message"]["content"]
            self.llama_finished.emit(result_text, self.chat_box)
        self.conservation.append({"role": "assistant", "content": result_text})
        self.llama_ended.emit(result_text)

class InputTextEdit(QTextEdit):
    enterPressed = pyqtSignal()
    escPressed = pyqtSignal()
    runPython = pyqtSignal()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Return and e.modifiers() == Qt.ShiftModifier:
            cursor = self.textCursor()
            cursor.insertText('\n')
        elif e.key() in (Qt.Key_Enter, Qt.Key_Return):
            if self.toPlainText() == "run":
                self.setPlainText("")
                self.runPython.emit()
            else:
                self.enterPressed.emit()
        elif e.key() == Qt.Key_Escape:
            self.escPressed.emit()
        else:
            super().keyPressEvent(e)

class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.conversation = []
        self.python_code = ""

    def initUI(self):
        # Set window properties
        self.setWindowTitle('mini llamma ui')
        self.setGeometry(600, 600, 800, 800)
        self.setStyleSheet("background-color: black; color: white;")

        self.message_input = InputTextEdit()
        self.message_input.setStyleSheet("background-color: black; color: white;")
        self.message_input.setAcceptRichText(False)

        self.message_input.setStyleSheet("font-size: 16px;")

        style_sheet_vertical = """
                    QScrollBar:vertical {
                        border: none;
                        background: transparent;
                        width: 8px;
                        margin: 0px;
                    }
                    QScrollBar::handle:vertical {
                                    background: rgba(51, 51, 51, 0.4);
                                    min-height: 20px;
                                }
                    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                        height: 0px;
                    }
                """

        style_sheet_horizontal = """
                    QScrollBar:horizontal {
                        border: none;
                        background: transparent;
                        height: 8px;
                        margin: 0px;
                    }
                    QScrollBar::handle:horizontal {
                        background: rgba(51, 51, 51, 0.4);
                        min-width: 20px;
                    }
                    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                        width: 0px;
                    }
                """
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.verticalScrollBar().setStyleSheet(style_sheet_vertical)
        scroll_area.horizontalScrollBar().setStyleSheet(style_sheet_horizontal)
        
        self.message_input.setPlaceholderText("type here...")

        # Create layout
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.addWidget(scroll_area)  # Add the scroll area to the main layout
        layout.addWidget(self.message_input)

        layout.setStretchFactor(scroll_area, 9)
        layout.setStretchFactor(self.message_input, 1)

        self.chat_layout = QVBoxLayout()
        self.chat_layout.setSpacing(5)
        self.chat_layout.setContentsMargins(3, 5, 3, 5)
        self.chat_layout.setAlignment(Qt.AlignTop)

        central_widget = QWidget()
        central_widget.setLayout(self.chat_layout)
        scroll_area.setWidget(central_widget)

        # Set layout
        self.setLayout(layout)

        self.message_input.enterPressed.connect(self.send_message)
        self.message_input.escPressed.connect(self.minimize_window)
        self.message_input.runPython.connect(self.run_python)
        QTimer.singleShot(0, self.set_message_input_focus)

    def set_message_input_focus(self):
        self.message_input.setFocus()

    def minimize_window(self):
        self.showMinimized()

    def create_markdown_widget(self) -> MarkdownTextEdit:
        # Create UI elements
        chat_box = MarkdownTextEdit()
        chat_box.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        chat_box.setFontPointSize(16)
        chat_box.setReadOnly(True)
        chat_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        chat_box.setContentsMargins(3, 5, 3, 5)  # Adjust these values as needed
        return chat_box

    def send_message(self):
        # Get message from input
        message = self.message_input.toPlainText()
        chat_box = self.create_markdown_widget()
        chat_box.setStyleSheet("color: #9370DB; font-size: 16px;")
        self.chat_layout.addWidget(chat_box)
        chat_box.setMarkdown(message)
        self.conversation.append({"role": "user", "content": message})
        self.message_input.clear()
        self.message_input.setDisabled(True)
        self.send_llama()

    def send_llama(self):
            chat_box = self.create_markdown_widget()
            # append to the new vlayout
            self.chat_layout.addWidget(chat_box)
            llama_worker = LlamaWorker(self.conversation, chat_box)
            llama_worker.llama_finished.connect(self.llama_finished)
            llama_worker.llama_ended.connect(self.llama_ended)
            llama_worker.setParent(self)

            llama_worker.start()
            self.process_events_timer = QTimer(self)
            self.process_events_timer.timeout.connect(QApplication.processEvents)
            self.process_events_timer.start(100)

    def llama_finished(self, result_text, chat_box):
        chat_box.setMarkdown(result_text)

    def llama_ended(self, output_text):
        # should probably add user confirmation
        self.message_input.setDisabled(False)
        if "```python\n" in output_text:
            chat_box = self.create_markdown_widget()
            self.chat_layout.addWidget(chat_box)
            chat_box.setMarkdown("Python code detected, type run to run the code")
            python_code = output_text.split('```python\n')[1].split("```")[0]
            self.python_code = python_code

    def run_python(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        exec(self.python_code)
        sys.stdout = sys.__stdout__
        output_result = captured_output.getvalue()
        chat_box = self.create_markdown_widget()
        self.chat_layout.addWidget(chat_box)
        chat_box.setMarkdown("Code Output:\n" + output_result)
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    chat_app = ChatApp()
    chat_app.show()
    sys.exit(app.exec_())
