#hsi_annotation/app.py
import sys

from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QApplication

from .ui.window import PaintWindow


def run(argv=None):
    # Reuse an existing QApplication when running inside Spyder / IPython
    # so the kernel does not die on a second run in the same session.
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    window = PaintWindow()
    window.show()
    exit_code = app.exec_()
    # Wait for all thread-pool workers (pixel spectrum reads) to finish
    # before the interpreter tears down Qt objects.
    QThreadPool.globalInstance().waitForDone(3000)
    return exit_code