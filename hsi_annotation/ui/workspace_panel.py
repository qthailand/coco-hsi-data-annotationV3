# hsi_annotation/ui/workspace_panel.py
"""
ui/workspace_panel.py
---------------------
File browser panel — lists .hdr files in a loaded directory.
Click a file to load it.
"""

import os
import logging

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton,
    QSizePolicy, QAbstractItemView,
)

log = logging.getLogger(__name__)


class WorkspacePanel(QWidget):
    """
    Signals:
        file_selected(path: str)  — emitted when user double-clicks or clicks + Load
        folder_opened(path: str)  — emitted when a folder is loaded
    """

    file_selected = pyqtSignal(str)
    folder_opened = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_dir = None
        self._hdr_files = []
        self._build_ui()

    def _build_ui(self):
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 8, 6, 6)
        root.setSpacing(4)

        header = QLabel("Workspace")
        header.setStyleSheet("font-weight: bold; font-size: 13px; color: #ccc;")
        root.addWidget(header)

        self._dir_label = QLabel("No folder loaded")
        self._dir_label.setStyleSheet(
            "font-size: 10px; color: #888; padding: 2px 0;")
        self._dir_label.setWordWrap(True)
        root.addWidget(self._dir_label)

        self._count_label = QLabel("")
        self._count_label.setStyleSheet(
            "font-size: 10px; color: #666; padding: 0;")
        root.addWidget(self._count_label)

        # File list
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._file_list.setAlternatingRowColors(True)
        self._file_list.setStyleSheet(
            "QListWidget {"
            "  border: 1px solid #444; border-radius: 4px;"
            "  background: #1e1e1e; color: #ddd;"
            "  alternate-background-color: #252525;"
            "  font-size: 11px;"
            "}"
            "QListWidget::item {"
            "  padding: 4px 6px;"
            "}"
            "QListWidget::item:selected {"
            "  background: #2d6ea4; color: white;"
            "}"
            "QListWidget::item:hover {"
            "  background: #333;"
            "}"
        )
        self._file_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._file_list.currentItemChanged.connect(self._on_current_changed)
        root.addWidget(self._file_list, stretch=1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._btn_open_folder = QPushButton("Open Folder")
        self._btn_open_folder.setToolTip("Browse for directory  [Ctrl+D]")
        self._btn_open_folder.setShortcut("Ctrl+D")
        self._btn_open_folder.clicked.connect(self._on_open_folder)

        self._btn_load = QPushButton("Load")
        self._btn_load.setToolTip("Load selected .hdr file")
        self._btn_load.setEnabled(False)
        self._btn_load.clicked.connect(self._on_load_clicked)

        btn_row.addWidget(self._btn_open_folder)
        btn_row.addWidget(self._btn_load)
        root.addLayout(btn_row)

        # Current loaded indicator
        self._loaded_label = QLabel("")
        self._loaded_label.setStyleSheet(
            "font-size: 10px; color: #5a9; padding: 2px 0;")
        self._loaded_label.setWordWrap(True)
        root.addWidget(self._loaded_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_directory(self, dir_path):
        """Scan directory for .hdr files and populate the list."""
        dir_path = os.path.abspath(dir_path)
        if not os.path.isdir(dir_path):
            log.warning("Not a directory: %s", dir_path)
            return

        self._current_dir = dir_path
        self._hdr_files = []

        for name in sorted(os.listdir(dir_path)):
            if name.lower().endswith(".hdr"):
                full = os.path.join(dir_path, name)
                if os.path.isfile(full):
                    self._hdr_files.append((name, full))

        self._file_list.clear()
        for name, full_path in self._hdr_files:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, full_path)
            item.setToolTip(full_path)
            self._file_list.addItem(item)

        self._dir_label.setText(dir_path)
        self._count_label.setText(
            "{} .hdr file{}".format(
                len(self._hdr_files), "" if len(self._hdr_files) == 1 else "s"))
        self._loaded_label.setText("")
        self._btn_load.setEnabled(False)

        self.folder_opened.emit(dir_path)
        log.info("Workspace loaded: %s  (%d .hdr files)",
                 dir_path, len(self._hdr_files))

    def mark_loaded(self, file_path):
        """Update the loaded-file indicator."""
        name = os.path.basename(file_path)
        self._loaded_label.setText(f"Loaded: {name}")

    def current_file_path(self):
        """Return the currently selected file path, or None."""
        item = self._file_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def select_file(self, file_path):
        """Programmatically select a file in the list."""
        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            if item.data(Qt.UserRole) == file_path:
                self._file_list.setCurrentItem(item)
                return

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_open_folder(self):
        from PyQt5.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory with .hdr files", self._current_dir or "")
        if dir_path:
            self.load_directory(dir_path)

    def _on_load_clicked(self):
        path = self.current_file_path()
        if path:
            self.file_selected.emit(path)

    def _on_item_double_clicked(self, item):
        path = item.data(Qt.UserRole)
        if path:
            self.file_selected.emit(path)

    def _on_current_changed(self, current, previous):
        self._btn_load.setEnabled(current is not None)