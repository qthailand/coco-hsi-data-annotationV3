# hsi_annotation/ui/category_panel.py
"""
ui/category_panel.py
--------------------
Left-side panel for managing annotation categories.

Columns:  ID (read-only) | Name (editable) | Color (click -> picker) | 👁

Signals:
    active_category_changed(category_id: int)
"""

import logging

from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QColorDialog, QLabel,
    QAbstractItemView, QSizePolicy, QCheckBox,
)

from ..registry import CategoryRegistry, AnnotationRegistry

log = logging.getLogger(__name__)

COL_ID      = 0
COL_NAME    = 1
COL_COLOR   = 2
COL_VISIBLE = 3
NUM_COLS    = 4


def _make_color_icon(color: QColor, w=40, h=18) -> QIcon:
    px = QPixmap(w, h)
    px.fill(color)
    return QIcon(px)


class _ColorButton(QPushButton):
    color_changed = pyqtSignal(QColor)

    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(QSize(48, 22))
        self.setFlat(False)
        self.setCursor(Qt.PointingHandCursor)
        self._refresh()
        self.clicked.connect(self._pick)

    def get_color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color: QColor):
        self._color = QColor(color)
        self._refresh()

    def _refresh(self):
        self.setIcon(_make_color_icon(self._color, 36, 14))
        self.setIconSize(QSize(36, 14))
        self.setStyleSheet(
            "QPushButton{{"
            "  border: 2px solid #666; border-radius: 3px;"
            "  background: {name};"
            "}}"
            "QPushButton:hover {{ border: 2px solid #ccc; }}"
            .format(name=self._color.name())
        )

    def _pick(self):
        c = QColorDialog.getColor(
            self._color, self, "Select category color",
            QColorDialog.ShowAlphaChannel,
        )
        if c.isValid():
            c.setAlpha(255)
            self._color = c
            self._refresh()
            self.color_changed.emit(c)


class CategoryPanel(QWidget):
    active_category_changed = pyqtSignal(int)

    def __init__(self, category_registry: CategoryRegistry,
                 annotation_registry: AnnotationRegistry,
                 parent=None):
        super().__init__(parent)
        self._categories = category_registry
        self._annotations = annotation_registry
        self._active_id = None
        self._updating = False

        self._build_ui()
        self._connect_registry()
        self._rebuild_table()

    def _build_ui(self):
        self.setMinimumWidth(230)
        self.setMaximumWidth(340)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 8, 6, 6)
        root.setSpacing(5)

        header = QLabel("Categories")
        header.setStyleSheet("font-weight: bold; font-size: 13px; color: #ccc;")
        root.addWidget(header)

        self._table = QTableWidget(0, NUM_COLS)
        self._table.setHorizontalHeaderLabels(["ID", "Name", "Color", "👁"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(COL_ID,      QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_NAME,    QHeaderView.Stretch)
        hh.setSectionResizeMode(COL_COLOR,   QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_VISIBLE, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
        )
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setStyleSheet(
            "QTableWidget {"
            "  border: 1px solid #444; border-radius: 4px;"
            "  background: #1e1e1e; color: #ddd;"
            "  alternate-background-color: #252525;"
            "}"
            "QTableWidget::item { padding: 2px 4px; }"
            "QTableWidget::item:selected {"
            "  background: #2d6ea4; color: white;"
            "}"
            "QHeaderView::section {"
            "  background: #2b2b2b; color: #bbb; padding: 4px;"
            "  border: none; border-bottom: 1px solid #444;"
            "}"
        )
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemChanged.connect(self._on_item_changed)
        root.addWidget(self._table)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self._btn_add = QPushButton("＋ Add")
        self._btn_add.setToolTip("Add category  [A]")
        self._btn_add.setShortcut("A")
        self._btn_add.clicked.connect(self.add_category)
        self._btn_del = QPushButton("－ Remove")
        self._btn_del.setToolTip("Remove selected category  [Del]")
        self._btn_del.clicked.connect(self.remove_selected)
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_del)
        root.addLayout(btn_row)

        self._active_bar = QLabel("Drawing as: —")
        self._active_bar.setAlignment(Qt.AlignCenter)
        self._active_bar.setWordWrap(True)
        self._active_bar.setStyleSheet(
            "QLabel {"
            "  background: #2d2d2d; border-radius: 4px;"
            "  padding: 5px; color: #ccc; font-size: 11px;"
            "}"
        )
        root.addWidget(self._active_bar)

    def _connect_registry(self):
        self._categories.category_added.connect(self._on_category_added)
        self._categories.category_removed.connect(self._on_category_removed)
        self._categories.category_changed.connect(self._on_category_changed)
        self._categories.reset.connect(self._rebuild_table)

    def _rebuild_table(self):
        self._updating = True
        self._table.clearContents()
        self._table.setRowCount(0)
        for cid in self._categories.ids():
            self._append_row(cid)
        self._updating = False

        if self._active_id and self._active_id in self._categories:
            self._select_row_for(self._active_id)
        elif self._categories.ids():
            self._select_row_for(self._categories.ids()[0])

    def _on_category_added(self, category_id):
        self._updating = True
        self._append_row(category_id)
        self._updating = False
        self._select_row_for(category_id)

    def _on_category_removed(self, category_id):
        row = self._row_for(category_id)
        if row < 0:
            return
        self._updating = True
        self._table.removeRow(row)
        self._updating = False

        n = self._table.rowCount()
        if n > 0:
            self._table.selectRow(min(row, n - 1))
        else:
            self._active_id = None
            self._active_bar.setText("Drawing as: —")
            self._active_bar.setStyleSheet(
                "QLabel { background:#2d2d2d; border-radius:4px;"
                " padding:5px; color:#ccc; font-size:11px; }"
            )

    def _on_category_changed(self, category_id):
        row = self._row_for(category_id)
        if row < 0:
            return
        self._updating = True
        name_item = self._table.item(row, COL_NAME)
        if name_item:
            name_item.setText(self._categories.name(category_id))
        btn = self._table.cellWidget(row, COL_COLOR)
        if btn:
            btn.set_color(self._categories.qcolor(category_id, alpha=255))
        self._updating = False
        if category_id == self._active_id:
            self._update_active_bar(category_id)

    def _on_selection_changed(self):
        if self._updating:
            return
        row = self._table.currentRow()
        if row < 0:
            return
        cid = self._category_id_at_row(row)
        if cid is None:
            return
        if cid != self._active_id:
            self._active_id = cid
            self._update_active_bar(cid)
            self.active_category_changed.emit(cid)

    def _on_item_changed(self, item):
        if self._updating:
            return
        if item.column() != COL_NAME:
            return
        cid = self._category_id_at_row(item.row())
        if cid is None:
            return
        new_name = item.text().strip()
        if not new_name:
            self._updating = True
            item.setText(self._categories.name(cid))
            self._updating = False
            return
        self._categories.set_name(cid, new_name)
        if cid == self._active_id:
            self._update_active_bar(cid)

    def _on_color_changed(self, category_id, color: QColor):
        self._categories.set_color(
            category_id, (color.red(), color.green(), color.blue()))
        if category_id == self._active_id:
            self._update_active_bar(category_id)

    def _on_visible_toggled(self, category_id, state):
        self._categories.set_visible(category_id, state == Qt.Checked)

    def _append_row(self, category_id):
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setRowHeight(row, 28)

        id_item = QTableWidgetItem(str(category_id))
        id_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        id_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, COL_ID, id_item)

        name_item = QTableWidgetItem(self._categories.name(category_id))
        name_item.setFlags(
            Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
        self._table.setItem(row, COL_NAME, name_item)

        btn = _ColorButton(self._categories.qcolor(category_id, alpha=255))
        btn.color_changed.connect(
            lambda c, cid=category_id: self._on_color_changed(cid, c))
        self._table.setCellWidget(row, COL_COLOR, btn)

        chk = QCheckBox()
        chk.setChecked(self._categories.is_visible(category_id))
        chk.setStyleSheet("QCheckBox { margin-left: 8px; }")
        chk.stateChanged.connect(
            lambda s, cid=category_id: self._on_visible_toggled(cid, s))
        self._table.setCellWidget(row, COL_VISIBLE, chk)

    def _row_for(self, category_id) -> int:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_ID)
            if item and int(item.text()) == category_id:
                return row
        return -1

    def _category_id_at_row(self, row) -> int:
        item = self._table.item(row, COL_ID)
        if item is None:
            return None
        try:
            return int(item.text())
        except ValueError:
            return None

    def _select_row_for(self, category_id):
        row = self._row_for(category_id)
        if row >= 0:
            self._table.selectRow(row)

    def _update_active_bar(self, category_id):
        name = self._categories.name(category_id)
        qc = self._categories.qcolor(category_id, alpha=255)
        fg = "#000" if qc.lightness() > 140 else "#fff"
        self._active_bar.setText(
            "Drawing as: <b>{}</b>  (ID {})".format(name, category_id))
        self._active_bar.setStyleSheet(
            "QLabel {{"
            "  background: {bg}; color: {fg};"
            "  border-radius: 4px; padding: 5px; font-size: 11px;"
            "}}".format(bg=qc.name(), fg=fg))

    # --- Public API ---

    def add_category(self):
        self._categories.add_category()

    def remove_selected(self):
        row = self._table.currentRow()
        if row < 0:
            return
        cid = self._category_id_at_row(row)
        if cid is None:
            return
        self._annotations.remove_by_category(cid)
        self._categories.remove_category(cid)

    def active_category_id(self):
        return self._active_id

    def active_color(self):
        if self._active_id:
            return self._categories.qcolor(self._active_id, alpha=255)
        return QColor(231, 76, 60)

    def active_name(self):
        return self._categories.name(self._active_id) if self._active_id else ""