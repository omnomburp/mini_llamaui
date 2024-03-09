import sys
import io
import ollama
import mistune
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit

class MarkdownTextEdit(QTextEdit):
    def setMarkdown(self, markdown_text):
        html_text = mistune.markdown(markdown_text)
        self.setHtml(html_text)

class LlamaWorker(QThread):
    llama_finished = pyqtSignal(str)
    llama_ended = pyqtSignal(str)

    def __init__(self, conversation):
        super().__init__()
        self.conservation = conversation

    def run(self):
        stream = ollama.chat(
            model='openhermes:7b-mistral-v2.5-q6_K',
            messages=self.conservation,
            stream=True,
        )

        result_text = ""
        for chunk in stream:
            result_text += chunk["message"]["content"]
            self.llama_finished.emit(result_text)
        self.conservation.append({"role": "assistant", "content": result_text})
        self.llama_ended.emit(result_text)

class InputTextEdit(QTextEdit):
    enterPressed = pyqtSignal()
    escPressed = pyqtSignal()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Enter, Qt.Key_Return):
            self.enterPressed.emit()
        elif e.key() == Qt.Key_Escape:
            self.escPressed.emit()
        else:
            super().keyPressEvent(e)

    def emitEnterPressed(self):
        self.enterPressed.emit()

class ChatApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.conversation = []
        self.f5_pressed = False
        self.f6_pressed = False

    def initUI(self):
        # Set window properties
        self.setWindowTitle('mini llamma ui')
        self.setGeometry(600, 600, 800, 800)
        self.setStyleSheet("background-color: black; color: white;")

        # Create UI elements
        self.chat_box = MarkdownTextEdit()
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

        self.chat_box.setStyleSheet("background-color: black; color: white;")
        self.chat_box.verticalScrollBar().setStyleSheet(style_sheet_vertical)
        self.chat_box.horizontalScrollBar().setStyleSheet(style_sheet_horizontal)
        self.chat_box.setReadOnly(True)

        self.message_input = InputTextEdit()
        self.message_input.setStyleSheet("background-color: black; color: white;")
        self.message_input.setAcceptRichText(False)

        self.chat_box.setFontPointSize(16)
        self.message_input.setStyleSheet("font-size: 16px;") # using this will make placeholder text bigger as well

        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.chat_box)
        layout.addWidget(self.message_input)

        layout.setStretchFactor(self.chat_box, 9)
        layout.setStretchFactor(self.message_input, 1)
        self.message_input.setPlaceholderText("type here...")
        self.message_input.setFocus()

        # Set layout
        self.setLayout(layout)

        self.message_input.enterPressed.connect(self.send_message)
        self.message_input.escPressed.connect(self.minimize_window)
        QTimer.singleShot(0, self.set_message_input_focus)

    def set_message_input_focus(self):
        self.message_input.setFocus()

    def minimize_window(self):
        self.showMinimized()

    def send_message(self):
        # Get message from input
        message = self.message_input.toPlainText()
        self.conversation.append({"role": "user", "content": message})

        self.message_input.clear()
        self.send_llama()

    def send_llama(self):
            llama_worker = LlamaWorker(self.conversation)
            llama_worker.llama_finished.connect(self.llama_finished)
            llama_worker.llama_ended.connect(self.llama_ended)
            llama_worker.setParent(self)

            llama_worker.start()
            self.process_events_timer = QTimer(self)
            self.process_events_timer.timeout.connect(QApplication.processEvents)
            self.process_events_timer.start(100)

    def llama_finished(self, result_text):
        self.chat_box.setMarkdown(result_text)
        scroll_bar = self.chat_box.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def llama_ended(self, output_text):
        # should probably add user confirmation
        if "```python\n" in output_text:
            self.chat_box.append("Python code detected")
            captured_output = io.StringIO()
            sys.stdout = captured_output
            python_code = output_text.split('```python\n')[1].split("```")[0]
            exec(python_code)
            sys.stdout = sys.__stdout__
            output_result = captured_output.getvalue()
            self.chat_box.append("Code Output: " + output_result)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    chat_app = ChatApp()
    chat_app.show()
    sys.exit(app.exec_())
