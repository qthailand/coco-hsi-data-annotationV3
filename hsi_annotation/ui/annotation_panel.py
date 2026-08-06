# hsi_annotation/ui/annotation_panel.py
"""
ui/annotation_panel.py
----------------------
Panel listing every annotation instance with a category dropdown.

Columns:  ID (read-only) | Category (QComboBox) | 👁 (visible toggle)

Signals:
    active_annotation_changed(annotation_id: int)   — None = deselected
    category_reassigned(annotation_id, old_cat, new_cat)
    delete_requested(annotation_id)
    new_annotation_requested()
"""

import logging

from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QCheckBox, QPushButton,
    QAbstractItemView, QSizePolicy,
)

from ..registry import CategoryRegistry, AnnotationRegistry

log = logging.getLogger(__name__)

COL_ID       = 0
COL_CATEGORY = 1
COL_VISIBLE  = 2
NUM_COLS     = 3


def _color_swatch_icon(color: QColor, w=14, h=14) -> QIcon:
    px = QPixmap(w, h)
    px.fill(color)
    return QIcon(px)


class AnnotationPanel(QWidget):
    active_annotation_changed = pyqtSignal(object)   # int or None
    category_reassigned       = pyqtSignal(int, int, int)
    delete_requested          = pyqtSignal(int)
    new_annotation_requested  = pyqtSignal()

    def __init__(self, category_registry: CategoryRegistry,
                 annotation_registry: AnnotationRegistry,
                 parent=None):
        super().__init__(parent)
        self._categories = category_registry
        self._annotations = annotation_registry
        self._updating = False
        self._combos = {}
        self._active_ann_id = None

        self._build_ui()
        self._connect_registries()
        self._rebuild_table()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 8, 6, 6)
        root.setSpacing(5)

        header = QLabel("Annotations")
        header.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #ccc;")
        root.addWidget(header)

        self._count_label = QLabel("0 annotations")
        self._count_label.setStyleSheet("font-size: 10px; color: #888;")
        root.addWidget(self._count_label)

        # Table
        self._table = QTableWidget(0, NUM_COLS)
        self._table.setHorizontalHeaderLabels(["ID", "Category", "👁"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(COL_ID,       QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_CATEGORY, QHeaderView.Stretch)
        hh.setSectionResizeMode(COL_VISIBLE,  QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
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
        root.addWidget(self._table)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self._btn_new = QPushButton("＋ New")
        self._btn_new.setToolTip("New annotation  [N]")
        self._btn_new.setShortcut("N")
        self._btn_new.clicked.connect(self._on_new_clicked)
        self._btn_del = QPushButton("－ Delete")
        self._btn_del.setToolTip("Delete selected annotation  [Del]")
        self._btn_del.clicked.connect(self._delete_selected)
        btn_row.addWidget(self._btn_new)
        btn_row.addWidget(self._btn_del)
        root.addLayout(btn_row)

        # Active annotation bar
        self._active_bar = QLabel("Select an annotation to draw")
        self._active_bar.setAlignment(Qt.AlignCenter)
        self._active_bar.setWordWrap(True)
        self._active_bar.setStyleSheet(
            "QLabel {"
            "  background: #2d2d2d; border-radius: 4px;"
            "  padding: 5px; color: #ccc; font-size: 11px;"
            "}"
        )
        root.addWidget(self._active_bar)

    # ------------------------------------------------------------------
    # Registry connections
    # ------------------------------------------------------------------

    def _connect_registries(self):
        self._annotations.annotation_added.connect(self._on_annotation_added)
        self._annotations.annotation_removed.connect(
            self._on_annotation_removed)
        self._annotations.annotation_changed.connect(
            self._on_annotation_changed)
        self._annotations.annotation_visibility_changed.connect(
            self._on_annotation_visibility_changed)
        self._annotations.reset.connect(self._on_reset)

        self._categories.category_added.connect(self._on_categories_mutated)
        self._categories.category_removed.connect(self._on_categories_mutated)
        self._categories.category_changed.connect(self._on_category_changed)
        self._categories.reset.connect(self._on_categories_mutated)

    # ------------------------------------------------------------------
    # Rebuild / sync
    # ------------------------------------------------------------------

    def _rebuild_table(self):
        self._updating = True
        self._combos.clear()
        self._table.clearContents()
        self._table.setRowCount(0)
        for ann in self._annotations.all():
            self._append_row(
                ann["id"], ann["category_id"], ann.get("visible", True))
        self._updating = False
        self._sync_count()

    def _on_reset(self):
        self._active_ann_id = None
        self._active_bar.setText("Select an annotation to draw")
        self._active_bar.setStyleSheet(
            "QLabel { background:#2d2d2d; border-radius:4px;"
            " padding:5px; color:#ccc; font-size:11px; }")
        self._rebuild_table()

    def _rebuild_all_combos(self):
        for ann_id, combo in self._combos.items():
            ann = self._annotations.get(ann_id)
            current_cat = ann["category_id"] if ann else 0
            self._populate_combo(combo, current_cat)

    def _populate_combo(self, combo, current_category_id):
        combo.blockSignals(True)
        combo.clear()
        for cid in self._categories.ids():
            name = self._categories.name(cid)
            qc = self._categories.qcolor(cid, alpha=255)
            combo.addItem(_color_swatch_icon(qc), name, cid)
            if cid == current_category_id:
                combo.setCurrentIndex(combo.count() - 1)
        combo.blockSignals(False)

    def _sync_count(self):
        n = len(self._annotations)
        self._count_label.setText(
            "{} annotation{}".format(n, "" if n == 1 else "s"))

    # ------------------------------------------------------------------
    # AnnotationRegistry handlers
    # ------------------------------------------------------------------

    def _on_annotation_added(self, ann_id):
        ann = self._annotations.get(ann_id)
        if ann is None:
            return
        self._updating = True
        self._append_row(
            ann_id, ann["category_id"], ann.get("visible", True))
        self._updating = False
        self._sync_count()
        # Auto-select newly created annotation
        self._select_row_for(ann_id)

    def _on_annotation_removed(self, ann_id):
        was_active = (ann_id == self._active_ann_id)
        row = self._row_for(ann_id)
        if row < 0:
            return
        self._updating = True
        self._table.removeRow(row)
        self._combos.pop(ann_id, None)
        self._updating = False
        self._sync_count()

        if was_active:
            self._active_ann_id = None
            n = self._table.rowCount()
            if n > 0:
                new_row = min(row, n - 1)
                self._table.selectRow(new_row)
            else:
                self._active_bar.setText("Select an annotation to draw")
                self._active_bar.setStyleSheet(
                    "QLabel { background:#2d2d2d; border-radius:4px;"
                    " padding:5px; color:#ccc; font-size:11px; }")
                self.active_annotation_changed.emit(None)

    def _on_annotation_changed(self, ann_id):
        if self._updating:
            return
        ann = self._annotations.get(ann_id)
        if ann is None:
            return
        combo = self._combos.get(ann_id)
        if combo is None:
            return
        cat_id = ann["category_id"]
        if combo.currentData() != cat_id:
            combo.blockSignals(True)
            for i in range(combo.count()):
                if combo.itemData(i) == cat_id:
                    combo.setCurrentIndex(i)
                    break
            combo.blockSignals(False)

    def _on_annotation_visibility_changed(self, ann_id):
        row = self._row_for(ann_id)
        if row < 0:
            return
        chk = self._table.cellWidget(row, COL_VISIBLE)
        if chk:
            self._updating = True
            chk.setChecked(self._annotations.is_visible(ann_id))
            self._updating = False

    # ------------------------------------------------------------------
    # CategoryRegistry handlers
    # ------------------------------------------------------------------

    def _on_categories_mutated(self, *_args):
        self._rebuild_all_combos()

    def _on_category_changed(self, category_id):
        for ann_id, combo in self._combos.items():
            for i in range(combo.count()):
                if combo.itemData(i) == category_id:
                    name = self._categories.name(category_id)
                    qc = self._categories.qcolor(category_id, alpha=255)
                    combo.blockSignals(True)
                    combo.setItemText(i, name)
                    combo.setItemIcon(i, _color_swatch_icon(qc))
                    combo.blockSignals(False)
                    break

    # ------------------------------------------------------------------
    # Row helpers
    # ------------------------------------------------------------------

    def _append_row(self, ann_id, category_id, visible):
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setRowHeight(row, 28)

        id_item = QTableWidgetItem(str(ann_id))
        id_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        id_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, COL_ID, id_item)

        combo = QComboBox()
        combo.setProperty("annotation_id", ann_id)
        self._populate_combo(combo, category_id)
        combo.currentIndexChanged.connect(
            lambda _idx, _combo=combo: self._on_combo_changed(_combo))
        self._table.setCellWidget(row, COL_CATEGORY, combo)
        self._combos[ann_id] = combo

        chk = QCheckBox()
        chk.setChecked(visible)
        chk.setStyleSheet("QCheckBox { margin-left: 8px; }")
        chk.stateChanged.connect(
            lambda state, _ann_id=ann_id: self._on_visible_toggled(
                _ann_id, state))
        self._table.setCellWidget(row, COL_VISIBLE, chk)

    def _row_for(self, ann_id) -> int:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_ID)
            if item and int(item.text()) == ann_id:
                return row
        return -1

    def _ann_id_at_row(self, row) -> int:
        item = self._table.item(row, COL_ID)
        if item is None:
            return None
        try:
            return int(item.text())
        except ValueError:
            return None

    def _select_row_for(self, ann_id):
        row = self._row_for(ann_id)
        if row >= 0:
            self._table.selectRow(row)

    def _update_active_bar(self, ann_id):
        if ann_id is None:
            self._active_bar.setText("Select an annotation to draw")
            self._active_bar.setStyleSheet(
                "QLabel { background:#2d2d2d; border-radius:4px;"
                " padding:5px; color:#ccc; font-size:11px; }")
            return
        ann = self._annotations.get(ann_id)
        if ann is None:
            return
        cat_id = ann["category_id"]
        cat_name = self._categories.name(cat_id)
        qc = self._categories.qcolor(cat_id, alpha=255)
        fg = "#000" if qc.lightness() > 140 else "#fff"
        self._active_bar.setText(
            "Drawing: <b>#{}</b> → {}".format(ann_id, cat_name))
        self._active_bar.setStyleSheet(
            "QLabel {{"
            "  background: {bg}; color: {fg};"
            "  border-radius: 4px; padding: 5px; font-size: 11px;"
            "}}".format(bg=qc.name(), fg=fg))

    # ------------------------------------------------------------------
    # User interaction
    # ------------------------------------------------------------------

    def _on_new_clicked(self):
        self.new_annotation_requested.emit()

    def _on_selection_changed(self):
        if self._updating:
            return
        row = self._table.currentRow()
        if row < 0:
            return
        ann_id = self._ann_id_at_row(row)
        if ann_id is None:
            return
        if ann_id != self._active_ann_id:
            self._active_ann_id = ann_id
            self._update_active_bar(ann_id)
            self.active_annotation_changed.emit(ann_id)

    def _on_combo_changed(self, combo):
        if self._updating:
            return
        ann_id = combo.property("annotation_id")
        new_cat_id = combo.currentData()
        if ann_id is None or new_cat_id is None:
            return
        ann = self._annotations.get(ann_id)
        if ann is None:
            return
        old_cat_id = ann["category_id"]
        if old_cat_id == new_cat_id:
            return
        self._annotations.update(ann_id, category_id=new_cat_id)
        self.category_reassigned.emit(ann_id, old_cat_id, new_cat_id)
        # Update active bar if this is the active annotation
        if ann_id == self._active_ann_id:
            self._update_active_bar(ann_id)

    def _on_visible_toggled(self, ann_id, state):
        if self._updating:
            return
        self._annotations.set_visible(ann_id, state == Qt.Checked)

    def _delete_selected(self):
        row = self._table.currentRow()
        if row < 0:
            return
        ann_id = self._ann_id_at_row(row)
        if ann_id is not None:
            self.delete_requested.emit(ann_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_annotation(self, ann_id):
        """Programmatically select an annotation."""
        self._select_row_for(ann_id)

    def active_annotation_id(self):
        return self._active_ann_id