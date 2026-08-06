# hsi_annotation/ui/window.py
from .pg_panel import PgPanel
from .paint_view import PaintView
from .contrast_dialog import ContrastDialog
from .category_panel import CategoryPanel
from .annotation_panel import AnnotationPanel
from .workspace_panel import WorkspacePanel
from ..data import (
    DEFAULT_HIGH_CUT,
    DEFAULT_LOW_CUT,
    build_category_id_mask,
    build_coco_annotation_json,
    compute_category_spectra,
)
from ..data import build_coco_annotations_from_mask
from ..canvas import BgItem, CanvasItem
from ..loader import list_hdr_files
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import (
    QColor, QFont, QImage, QPixmap, QBrush,
    QPainter, QPolygonF, QPen, QIcon,
)
from PyQt5.QtCore import (
    QObject, QThread, QRectF, Qt, pyqtSignal, pyqtSlot, QPointF,
)
import numpy as np
import os
import json
import logging
from ..registry import CategoryRegistry, AnnotationRegistry, _DEFAULT_COLORS

log = logging.getLogger(__name__)


class CategorySpectrumWorker(QObject):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    @pyqtSlot(object, object, object)
    def process(self, datacube, mask, categories):
        try:
            self.finished.emit(
                compute_category_spectra(datacube, mask, categories))
        except Exception as exc:
            log.exception("CategorySpectrumWorker error")
            self.error.emit(str(exc))


class PaintWindow(QMainWindow):
    category_spectra_request = pyqtSignal(object, object, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI Ground Truth Painter")
        self.resize(1500, 900)

        self._preview_low_cut = DEFAULT_LOW_CUT
        self._preview_high_cut = DEFAULT_HIGH_CUT
        self._current_hdr_path = None
        self._cube_ann_cache = {}  # {hdr_path: {"annotations": [...], "next_id": int}}
        self._cube_rgb_cache = {}  # {hdr_path: QImage}
        # --- Registries ---
        self._cat_registry   = CategoryRegistry(self)
        self._annot_registry = AnnotationRegistry(self)

        self._cat_registry.add_category("Category 1", (231,  76,  60))
        self._cat_registry.add_category("Category 2", ( 46, 204, 113))

        # --- Canvas / scene ---
        self._scene  = QGraphicsScene(self)
        self._bg     = BgItem()
        self._canvas = CanvasItem()
        self._scene.addItem(self._bg)
        self._scene.addItem(self._canvas)
        self._scene.setSceneRect(QRectF(0, 0, 800, 600))

        self._view     = PaintView(self._scene, self)
        self._pg_panel = PgPanel(self)

        # --- Left column: Workspace + Category + Spectrum ---
        self._workspace_panel  = WorkspacePanel(self)
        self._annotation_panel = AnnotationPanel(
            self._cat_registry, self._annot_registry, self)
        self._category_panel = CategoryPanel(
            self._cat_registry, self._annot_registry, self)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self._workspace_panel,  stretch=2)
        left_layout.addWidget(self._category_panel,   stretch=2)
        left_layout.addWidget(self._pg_panel,         stretch=3)

        # --- Main splitter: Left | Canvas | AnnotationPanel ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self._view)
        splitter.addWidget(self._annotation_panel)
        splitter.setSizes([280, 900, 280])
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(5)
        self.setCentralWidget(splitter)

        # --- Toolbar / statusbar ---
        self._tool_actions = {}
        self._bbox_items   = []
        self._build_toolbar()
        self._build_statusbar()

        # --- Workspace signals ---
        self._workspace_panel.file_selected.connect(self._on_workspace_file_selected)

        # --- Annotation panel signals ---
        self._annotation_panel.new_annotation_requested.connect(
            self._create_new_annotation)
        self._annotation_panel.active_annotation_changed.connect(
            self._on_active_annotation_changed)
        self._annotation_panel.category_reassigned.connect(
            self._on_annotation_category_reassigned)
        self._annotation_panel.delete_requested.connect(
            self._on_annotation_delete_requested)
        self._annot_registry.annotation_visibility_changed.connect(
            self._on_annotation_visibility_changed)

        # --- Canvas signals ---
        self._canvas.signals.updated.connect(self._refresh_pg)
        self._canvas.signals.shape_closed.connect(self._on_shape_closed)
        self._canvas.signals.shape_closed.connect(
            self._compute_category_spectra)
        self._canvas.signals.shape_closed.connect(
            self._update_bbox_overlays)
        self._canvas.signals.shape_completed.connect(
            self._on_shape_completed)
        self._canvas.signals.spectrum_ready.connect(
            self._pg_panel.update_spectrum)
        self._canvas.signals.loaded.connect(self._on_loaded)

        # --- Category registry → canvas sync ---
        self._cat_registry.category_changed.connect(
            self._on_registry_category_changed)
        self._cat_registry.category_color_changed.connect(
            self._on_registry_category_color_changed)
        self._cat_registry.category_about_to_be_removed.connect(
            self._on_category_about_to_be_removed)
        self._cat_registry.category_removed.connect(
            self._on_registry_category_removed)

        # --- Spectrum worker thread ---
        self._spectra_running = False
        self._spectra_pending = False
        self._spectrum_thread = QThread(self)
        self._spectrum_worker = CategorySpectrumWorker()
        self._spectrum_worker.moveToThread(self._spectrum_thread)
        self._spectrum_worker.finished.connect(
            self._on_category_spectra_ready)
        self._spectrum_worker.error.connect(
            self._on_category_spectra_error)
        self.category_spectra_request.connect(
            self._spectrum_worker.process)
        self._spectrum_thread.start()

    # ------------------------------------------------------------------
    # Workspace → load datacube
    # ------------------------------------------------------------------

    def _on_workspace_file_selected(self, hdr_path):
        if hdr_path == self._current_hdr_path:
            return

        # Save current cube's annotations before switching
        if self._canvas.is_loaded and self._current_hdr_path:
            self._save_cube_state()

        self._clear_and_load(hdr_path)

    def _clear_and_load(self, hdr_path):
        """Clear current view and load a new datacube."""
        self._annot_registry.clear()
        self._canvas.clear_mask()
        self._canvas.set_current_annotation(None)

        log.info("Loading from workspace: %s", hdr_path)
        self.statusBar().showMessage(
            f"Loading {os.path.basename(hdr_path)}...")
        QApplication.processEvents()

        # Set path BEFORE load so _on_loaded uses correct key
        self._current_hdr_path = hdr_path

        self._canvas.set_preview_cuts(
            self._preview_low_cut, self._preview_high_cut)
        rgb_img = self._canvas.load_datacube(hdr_path)
        self._bg.setPixmap(QPixmap.fromImage(rgb_img))
        self._scene.setSceneRect(QRectF(rgb_img.rect()))

        # Restore annotations: from cache or auto-load GT
        self._restore_cube_state(hdr_path)

        self._workspace_panel.mark_loaded(hdr_path)

    def _save_cube_state(self):
        """Cache current cube's annotations AND layer pixels."""
        if not self._current_hdr_path:
            return

        layer_pixels = {}
        for ann_id in list(self._canvas._layers.keys()):
            layer_np = self._canvas.layer_to_numpy(ann_id)
            if layer_np is not None:
                layer_pixels[ann_id] = layer_np

        self._cube_ann_cache[self._current_hdr_path] = {
            "registry": self._annot_registry.to_state(),
            "layer_pixels": layer_pixels,
            "layer_visible": dict(self._canvas._layer_visible),
            "width":  self._canvas._layer_w,
            "height": self._canvas._layer_h,
        }

    def _restore_cube_state(self, hdr_path):
        """Restore annotations and layers from cache, or auto-load GT."""
        cached = self._cube_ann_cache.get(hdr_path)
        if cached:
            # Restore registry
            self._annot_registry.from_state(cached["registry"])

            layer_pixels = cached.get("layer_pixels", {})
            layer_visible = cached.get("layer_visible", {})

            self._canvas._layers.clear()
            self._canvas._layer_visible.clear()
            w, h = self._canvas._layer_w, self._canvas._layer_h

            if layer_pixels:
                # ── Restore from cached pixel data ──
                for ann_id, layer_np in layer_pixels.items():
                    ann_id = int(ann_id)
                    img = QImage(
                        np.ascontiguousarray(layer_np).data,
                        w, h, w * 4, QImage.Format_RGBA8888
                    ).copy().convertToFormat(QImage.Format_ARGB32)
                    self._canvas._layers[ann_id] = img

                for ann_id, vis in layer_visible.items():
                    self._canvas._layer_visible[int(ann_id)] = vis

                log.info("Restored %s: %d anns from cached pixels",
                         os.path.basename(hdr_path),
                         len(self._canvas._layers))
            else:
                # ── No cached pixels — rebuild layers from segmentation ──
                for ann in self._annot_registry.all():
                    self._create_layer_for_annotation(ann)
                log.info("Rebuilt %s: %d layers from segmentation",
                         os.path.basename(hdr_path),
                         len(self._canvas._layers))

            self._canvas._composite()

            all_anns = self._annot_registry.all()
            if all_anns:
                self._annotation_panel.select_annotation(
                    all_anns[0]["id"])
            self._refresh_pg()
            self._update_bbox_overlays()
            self._compute_category_spectra()
        else:
            # First time — try auto-load GT
            self._auto_load_gt(hdr_path)

    # def _auto_load_gt(self, hdr_path):
    #     """Look for workspace-level or per-cube GT."""
    #     parent = os.path.dirname(hdr_path)

    #     # Priority 1: workspace-level annotations.json
    #     workspace_json = os.path.join(parent, "annotations.json")
    #     if os.path.isfile(workspace_json):
    #         log.info("Auto-loading workspace GT: %s", workspace_json)
    #         self._load_workspace_gt(workspace_json)
    #         return

    #     # Priority 2: per-cube JSON next to .hdr
    #     stem = os.path.splitext(hdr_path)[0]
    #     candidates = [
    #         f"{stem}.json",
    #         f"{stem}_gt.json",
    #         f"{stem}_coco.json",
    #     ]
    #     gt_dir = os.path.join(parent, "GT")
    #     if os.path.isdir(gt_dir):
    #         base = os.path.basename(stem)
    #         for name in os.listdir(gt_dir):
    #             if name.lower().endswith(".json") and base in name:
    #                 candidates.append(os.path.join(gt_dir, name))

    #     for json_path in candidates:
    #         if os.path.isfile(json_path):
    #             log.info("Auto-loading cube GT: %s", json_path)
    #             self._load_coco_from_path(json_path)
    #             return

    #     log.debug("No existing GT for %s", hdr_path)
    def _auto_load_gt(self, hdr_path):
        """Look for per-cube GT JSON."""
        stem = os.path.splitext(hdr_path)[0]
        for ext in [".bsq.hdr", ".hdr", ".HDR"]:
            if stem.lower().endswith(ext.lower()):
                stem = stem[:len(stem) - len(ext)]
                break

        candidates = [
            f"{stem}.json",
            f"{stem}_gt.json",
        ]

        for json_path in candidates:
            if os.path.isfile(json_path):
                log.info("Auto-loading GT: %s", json_path)
                self._load_coco_from_path(json_path)
                return

        log.debug("No existing GT for %s", hdr_path)
    # def _load_coco_from_path(self, json_path, replace=True,
    #                          merge_categories=True):
    #     """Load COCO JSON for current cube.
    #     replace=True   → clear annotations + layers first
    #     replace=False  → add to existing
    #     merge_categories=True  → add missing categories, keep existing
    #     merge_categories=False → replace categories entirely
    #     """
    #     try:
    #         with open(json_path, "r", encoding="utf-8") as fh:
    #             coco = json.load(fh)
    #     except Exception as exc:
    #         log.warning("Failed to load GT %s: %s", json_path, exc)
    #         return

    #     images = coco.get("images", [])
    #     if not images:
    #         log.warning("No images entry in COCO JSON")
    #         return

    #     categories = coco.get("categories", [])
    #     anns = coco.get("annotations", [])

    #     # ── Categories ──
    #     if replace or not merge_categories:
    #         self._cat_registry.clear()
    #         cat_data = {}
    #         for idx, cat in enumerate(categories):
    #             cid   = int(cat.get("id", 0))
    #             name  = str(cat.get("name", f"Category {cid}"))
    #             color = _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
    #             cat_data[cid] = {
    #                 "name": name, "color": color, "visible": True}
    #         self._cat_registry.load(
    #             {str(k): v for k, v in cat_data.items()})
    #     else:
    #         # Add missing categories only
    #         existing_names = {
    #             self._cat_registry.name(cid): cid
    #             for cid in self._cat_registry.ids()
    #         }
    #         for cat in categories:
    #             cat_name = str(cat.get("name", ""))
    #             if cat_name not in existing_names:
    #                 color = _DEFAULT_COLORS[
    #                     len(self._cat_registry) % len(_DEFAULT_COLORS)]
    #                 self._cat_registry.add_category(cat_name, color)

    #     # Build name→id map for category remapping
    #     name_to_id = {
    #         self._cat_registry.name(cid): cid
    #         for cid in self._cat_registry.ids()
    #     }

    #     # ── Clear annotations + layers if replacing ──
    #     if replace:
    #         # Remove all layers
    #         for ann in list(self._annot_registry.all()):
    #             self._canvas.remove_layer(ann["id"])
    #         self._annot_registry.clear()

    #     # ── Annotations ──
    #     for a in anns:
    #         seg = a.get("segmentation", [])
    #         seg_bbox = None
    #         if seg:
    #             flat_pts = []
    #             for poly in seg:
    #                 it = iter(poly)
    #                 for x, y in zip(it, it):
    #                     flat_pts.append((float(x), float(y)))
    #             if flat_pts:
    #                 xs = [p[0] for p in flat_pts]
    #                 ys = [p[1] for p in flat_pts]
    #                 xm, ym = min(xs), min(ys)
    #                 seg_bbox = [xm, ym,
    #                             max(xs) - xm, max(ys) - ym]
    #         raw_bbox = a.get("bbox", [])
    #         use_bbox = (seg_bbox if seg_bbox
    #                     else (raw_bbox if len(raw_bbox) >= 4
    #                           else []))

    #         # Remap category_id by name
    #         coco_cat_id = int(a.get("category_id", 0))
    #         cat_name = None
    #         for cat in categories:
    #             if int(cat.get("id", 0)) == coco_cat_id:
    #                 cat_name = cat.get("name")
    #                 break
    #         local_cat_id = name_to_id.get(cat_name, coco_cat_id)

    #         # Get color for this category
    #         cat_color_rgb = self._cat_registry.color(local_cat_id)

    #         # Add to registry
    #         ann_id = self._annot_registry.add(
    #             local_cat_id,
    #             bbox=use_bbox,
    #             segmentation=seg,
    #             area=int(a.get("area", 0)),
    #         )
    #         # Force update visible
    #         ann_entry = self._annot_registry.get(ann_id)
    #         if ann_entry:
    #             ann_entry["visible"] = True

    #         # Create layer
    #         if seg:
    #             polygons = seg if isinstance(seg[0], list) else [seg]
    #             self._canvas.paint_annotation_layer(
    #                 ann_id, polygons, cat_color_rgb)
    #         elif use_bbox:
    #             x0, y0, bw, bh = use_bbox[:4]
    #             poly = [float(x0), float(y0),
    #                     float(x0+bw), float(y0),
    #                     float(x0+bw), float(y0+bh),
    #                     float(x0), float(y0+bh)]
    #             self._canvas.paint_annotation_layer(
    #                 ann_id, [poly], cat_color_rgb)

    #     # Composite
    #     self._canvas._composite()

    #     # Select first annotation
    #     all_anns = self._annot_registry.all()
    #     if all_anns:
    #         self._annotation_panel.select_annotation(all_anns[0]["id"])

    #     self._refresh_pg()
    #     self._update_bbox_overlays()
    #     self._compute_category_spectra()

    #     self.statusBar().showMessage(
    #         "Loaded: {} annotations, {} categories".format(
    #             len(anns), len(categories)), 3000)
    def _load_coco_from_path(self, json_path, replace=True):
        """Load COCO JSON for current cube."""
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                coco = json.load(fh)
        except Exception as exc:
            log.warning("Failed to load GT %s: %s", json_path, exc)
            return

        images     = coco.get("images", [])
        anns       = coco.get("annotations", [])
        categories = coco.get("categories", [])

        if not images:
            log.warning("No images in COCO JSON")
            return

        # ── Categories ──
        self._cat_registry.clear()
        for idx, cat in enumerate(categories):
            cid   = int(cat.get("id", 0))
            name  = str(cat.get("name", f"Category {cid}"))
            color = _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
            self._cat_registry.add_category(name, color)

        name_to_id = {
            self._cat_registry.name(cid): cid
            for cid in self._cat_registry.ids()
        }

        # ── Clear old annotations + layers ──
        for ann in list(self._annot_registry.all()):
            self._canvas.remove_layer(ann["id"])
        self._annot_registry.clear()

        # ── Load annotations ──
        for a in anns:
            seg = a.get("segmentation", [])
            seg_bbox = None
            if seg:
                flat_pts = []
                for poly in seg:
                    it = iter(poly)
                    for x, y in zip(it, it):
                        flat_pts.append((float(x), float(y)))
                if flat_pts:
                    xs = [p[0] for p in flat_pts]
                    ys = [p[1] for p in flat_pts]
                    xm, ym = min(xs), min(ys)
                    seg_bbox = [xm, ym,
                                max(xs) - xm, max(ys) - ym]
            raw_bbox = a.get("bbox", [])
            use_bbox = (seg_bbox if seg_bbox
                        else (raw_bbox if len(raw_bbox) >= 4 else []))

            # Remap category
            coco_cat_id = int(a.get("category_id", 0))
            cat_name = None
            for cat in categories:
                if int(cat.get("id", 0)) == coco_cat_id:
                    cat_name = cat.get("name")
                    break
            local_cat_id = name_to_id.get(cat_name, coco_cat_id)
            color_rgb = self._cat_registry.color(local_cat_id)

            ann_id = self._annot_registry.add(
                local_cat_id,
                bbox=use_bbox,
                segmentation=seg,
                area=int(a.get("area", 0)),
            )

            # Create layer
            if seg:
                polygons = seg if isinstance(seg[0], list) else [seg]
                self._canvas.paint_annotation_layer(
                    ann_id, polygons, color_rgb)
            elif use_bbox:
                x0, y0, bw, bh = use_bbox[:4]
                poly = [float(x0), float(y0),
                        float(x0+bw), float(y0),
                        float(x0+bw), float(y0+bh),
                        float(x0), float(y0+bh)]
                self._canvas.paint_annotation_layer(
                    ann_id, [poly], color_rgb)

        self._canvas._composite()

        all_anns = self._annot_registry.all()
        if all_anns:
            self._annotation_panel.select_annotation(all_anns[0]["id"])
        self._refresh_pg()
        self._update_bbox_overlays()
        self._compute_category_spectra()

        self.statusBar().showMessage(
            "Loaded: {} annotations, {} categories".format(
                len(anns), len(categories)), 3000)
    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    @staticmethod
    def _make_probe_icon():
        px = QPixmap(20, 20)
        px.fill(Qt.transparent)
        p = QPainter(px)
        p.setRenderHint(QPainter.Antialiasing)
        colors = [
            QColor(255, 50, 50),
            QColor(255, 200, 0),
            QColor(50, 255, 50),
            QColor(50, 150, 255),
            QColor(180, 50, 255),
        ]
        bar_w = 3
        for i, c in enumerate(colors):
            p.fillRect(1 + i * bar_w, 2, bar_w, 16, c)
        p.end()
        return QIcon(px)

    def _build_toolbar(self):
        tb = QToolBar("Tools", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        # Probe button
        probe_act = QAction(self._make_probe_icon(), "Probe", self)
        probe_act.setCheckable(True)
        probe_act.setToolTip("Click-to-probe spectrum  [P]")
        probe_act.setShortcut("P")
        probe_act.triggered.connect(lambda _: self._set_tool("probe"))
        tb.addAction(probe_act)
        self._tool_actions["probe"] = probe_act
        tb.addSeparator()

        # Drawing tools
        for tool_key, label, shortcut in [
            ("connect", "Draw", "L"),
            ("circle",  "Circle",  "C"),
            ("eraser",  "Eraser",  "E"),
        ]:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setShortcut(shortcut)
            act.triggered.connect(lambda _, t=tool_key: self._set_tool(t))
            tb.addAction(act)
            self._tool_actions[tool_key] = act

        self._tool_actions["connect"].setChecked(True)
        tb.addSeparator()

        tb.addWidget(QLabel("  Size: "))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(1, 100)
        self._size_spin.setValue(4)
        self._size_spin.setFixedWidth(60)
        self._size_spin.valueChanged.connect(self._canvas.set_pen_width)
        tb.addWidget(self._size_spin)

        tb.addWidget(QLabel("  Opacity: "))
        opacity_spin = QSpinBox()
        opacity_spin.setRange(10, 100)
        opacity_spin.setValue(75)
        opacity_spin.setSuffix(" %")
        opacity_spin.setFixedWidth(70)
        opacity_spin.valueChanged.connect(
            lambda v: self._canvas.setOpacity(v / 100))
        tb.addWidget(opacity_spin)
        tb.addSeparator()

        tb.addWidget(QLabel("  Zoom: "))
        for label, shortcut, slot in [
            ("+",  "Ctrl+=", self._view.zoom_in),
            ("-",  "Ctrl+-", self._view.zoom_out),
            ("1:1", "Ctrl+0", self._view.zoom_reset),
            ("Fit", "Ctrl+F", self._fit),
        ]:
            act = QAction(label, self)
            act.setShortcut(shortcut)
            act.triggered.connect(slot)
            tb.addAction(act)
        tb.addSeparator()

        self._contrast_action = QAction("RGB Contrast", self)
        self._contrast_action.triggered.connect(self._open_contrast_dialog)
        self._contrast_action.setEnabled(False)
        tb.addAction(self._contrast_action)
        tb.addSeparator()

        open_folder_act = QAction("Open Folder", self)
        open_folder_act.setShortcut("Ctrl+O")
        open_folder_act.triggered.connect(self._open_folder)
        tb.addAction(open_folder_act)

        clear_act = QAction("Clear GT", self)
        clear_act.setShortcut("Ctrl+N")
        clear_act.triggered.connect(self._clear)
        tb.addAction(clear_act)

        save_act = QAction("Save GT", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save)
        tb.addAction(save_act)

        load_gt_act = QAction("Load GT", self)
        load_gt_act.setShortcut("Ctrl+L")
        load_gt_act.triggered.connect(self._load_gt)
        tb.addAction(load_gt_act)

    def _build_statusbar(self):
        self._zoom_label = QLabel("Zoom: 100%")
        self.statusBar().addPermanentWidget(self._zoom_label)
        for action in self._tool_actions.values():
            action.setEnabled(False)
        self.statusBar().showMessage(
            "Open a folder (Ctrl+D) or .hdr file (Ctrl+O)")

    # ------------------------------------------------------------------
    # Tool / view helpers
    # ------------------------------------------------------------------

    def update_zoom_label(self, zoom):
        self._zoom_label.setText("Zoom: {:.0f}%".format(zoom * 100))

    def _set_tool(self, tool_name):
        log.debug("Active tool: %s", tool_name)
        self._canvas.set_tool(tool_name)
        for name, action in self._tool_actions.items():
            action.setChecked(name == tool_name)

    def _fit(self):
        self._view.fitInView(
            self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._view._zoom = self._view.transform().m11()
        self.update_zoom_label(self._view._zoom)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _open_single(self):
        """Open a single .hdr file (legacy behavior)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Hyperspectral Datacube", "",
            "ENVI Header (*.hdr)")
        if not path:
            return
        # Also populate workspace with sibling .hdr files
        parent_dir = os.path.dirname(path)
        self._workspace_panel.load_directory(parent_dir)
        self._workspace_panel.select_file(path)
        self._clear_and_load(path)

    def _open_folder(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory with .hdr files", "")
        if not dir_path:
            return

        # Save current cube before switching folder
        if self._canvas.is_loaded and self._current_hdr_path:
            self._save_cube_state()

        self._workspace_panel.load_directory(dir_path)

        hdr_files = list_hdr_files(dir_path)
        if hdr_files:
            self._workspace_panel.select_file(hdr_files[0])
            self._clear_and_load(hdr_files[0])
        else:
            QMessageBox.information(
                self, "No .hdr files",
                f"No .hdr files found in:\n{dir_path}")

    def _load_gt(self):
        if not self._canvas.is_loaded:
            QMessageBox.warning(self, "No data", "Load a datacube first.")
            return

        hdr_path = self._current_hdr_path
        stem = os.path.splitext(hdr_path)[0]
        for ext in [".bsq.hdr", ".hdr", ".HDR"]:
            if stem.lower().endswith(ext.lower()):
                stem = stem[:len(stem) - len(ext)]
                break

        default_dir = os.path.dirname(hdr_path)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Annotations", default_dir,
            "COCO JSON (*.json)")
        if not path:
            return

        self._load_coco_from_path(path)

    # def _load_gt(self):
    #     path, _ = QFileDialog.getOpenFileName(
    #         self, "Load Annotations", "",
    #         "COCO JSON (*.json)")
    #     if not path:
    #         return
    #     self._load_workspace_gt(path)

    def _load_workspace_gt(self, json_path):
        """Load a workspace-level COCO JSON containing all cubes."""
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                coco = json.load(fh)
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return

        images      = coco.get("images", [])
        anns        = coco.get("annotations", [])
        categories  = coco.get("categories", [])

        if not images:
            QMessageBox.warning(self, "Invalid", "No images in JSON")
            return

        # ── Load categories ──
        self._cat_registry.clear()
        cat_data = {}
        for idx, cat in enumerate(categories):
            cid   = int(cat.get("id", 0))
            name  = str(cat.get("name", f"Category {cid}"))
            color = _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
            cat_data[cid] = {"name": name, "color": color,
                             "visible": True}
        self._cat_registry.load(
            {str(k): v for k, v in cat_data.items()})

        # ── Group annotations by image_id ──
        anns_by_image = {}
        for a in anns:
            img_id = int(a.get("image_id", 0))
            anns_by_image.setdefault(img_id, []).append(a)

        # ── Resolve image file_name → hdr_path ──
        workspace_dir = self._workspace_panel._current_dir or \
            os.path.dirname(json_path)

        for img_entry in images:
            img_id   = int(img_entry.get("id", 0))
            fname    = img_entry.get("file_name", "")
            hdr_path = os.path.join(workspace_dir, fname) \
                if fname and not os.path.isabs(fname) else fname

            if not os.path.isfile(hdr_path):
                log.warning("Cannot find cube: %s", hdr_path)
                continue

            # Build annotation list for this cube
            cube_anns = anns_by_image.get(img_id, [])
            name_to_id = {
                self._cat_registry.name(cid): cid
                for cid in self._cat_registry.ids()
            }

            ann_list = []
            for a in cube_anns:
                seg = a.get("segmentation", [])
                seg_bbox = None
                if seg:
                    flat_pts = []
                    for poly in seg:
                        it = iter(poly)
                        for x, y in zip(it, it):
                            flat_pts.append((float(x), float(y)))
                    if flat_pts:
                        xs = [p[0] for p in flat_pts]
                        ys = [p[1] for p in flat_pts]
                        xm, ym = min(xs), min(ys)
                        seg_bbox = [xm, ym,
                                    max(xs) - xm, max(ys) - ym]
                raw_bbox = a.get("bbox", [])
                use_bbox = (seg_bbox if seg_bbox
                            else (raw_bbox if len(raw_bbox) >= 4
                                  else []))

                coco_cat_id = int(a.get("category_id", 0))
                cat_name = None
                for cat in categories:
                    if int(cat.get("id", 0)) == coco_cat_id:
                        cat_name = cat.get("name")
                        break
                local_cat_id = name_to_id.get(
                    cat_name, coco_cat_id)

                ann_list.append({
                    "id":           int(a.get("id", 0)),
                    "category_id":  local_cat_id,
                    "bbox":         use_bbox,
                    "segmentation": seg,
                    "area":         int(a.get("area", 0)),
                    "visible":      True,
                })

            # Store in cache
            from ..registry import AnnotationRegistry
            state = {
                "annotations": ann_list,
                "next_id": max(
                    (a["id"] for a in ann_list), default=0) + 1,
            }
            self._cube_ann_cache[hdr_path] = {
                "registry": state,
                "layer_pixels": {},
                "layer_visible": {},
            }
            log.info("Cached annotations for %s: %d",
                     fname, len(ann_list))

        # ── Load current cube if it has annotations ──
        if self._current_hdr_path:
            cached = self._cube_ann_cache.get(self._current_hdr_path)
            if cached:
                self._annot_registry.from_state(cached["registry"])
                # Create layers
                for ann in self._annot_registry.all():
                    self._create_layer_for_annotation(ann)
                self._canvas._composite()

                all_anns = self._annot_registry.all()
                if all_anns:
                    self._annotation_panel.select_annotation(
                        all_anns[0]["id"])
                self._refresh_pg()
                self._update_bbox_overlays()
                self._compute_category_spectra()

        self.statusBar().showMessage(
            "Loaded: {} cubes, {} annotations, {} categories".format(
                len(images), len(anns), len(categories)), 3000)

    def _create_layer_for_annotation(self, ann):
        """Create a canvas layer for an annotation."""
        ann_id   = ann["id"]
        cat_id   = ann["category_id"]
        color    = self._cat_registry.color(cat_id)
        seg      = ann.get("segmentation", [])
        if seg:
            polygons = seg if isinstance(seg[0], list) else [seg]
            self._canvas.paint_annotation_layer(
                ann_id, polygons, color)
        else:
            bbox = ann.get("bbox", [])
            if len(bbox) >= 4:
                x0, y0, bw, bh = bbox[:4]
                poly = [float(x0), float(y0),
                        float(x0+bw), float(y0),
                        float(x0+bw), float(y0+bh),
                        float(x0), float(y0+bh)]
                self._canvas.paint_annotation_layer(
                    ann_id, [poly], color)

    def _open_contrast_dialog(self):
        if not self._canvas.is_loaded:
            return
        orig_low, orig_high = (
            self._preview_low_cut, self._preview_high_cut)
        dialog = ContrastDialog(orig_low, orig_high, self)
        dialog.preview_changed.connect(self._apply_preview_cuts)
        if dialog.exec_() != QDialog.Accepted:
            self._apply_preview_cuts(orig_low, orig_high)
            self._preview_low_cut = orig_low
            self._preview_high_cut = orig_high
            return
        self._preview_low_cut, self._preview_high_cut = dialog.values()
        self.statusBar().showMessage(
            "Contrast updated  |  {}".format(
                self._format_preview_info()), 5000)

    def _apply_preview_cuts(self, low_cut, high_cut):
        rgb_img = self._canvas.render_preview(low_cut, high_cut)
        if rgb_img:
            self._bg.setPixmap(QPixmap.fromImage(rgb_img))

    def _clear(self):
        if not self._canvas.is_loaded:
            return
        if QMessageBox.question(
            self, "Clear GT", "Clear all annotations and mask?",
            QMessageBox.Yes | QMessageBox.No,
        ) == QMessageBox.Yes:
            self._annot_registry.clear()
            self._canvas.clear_mask()
            self._canvas.set_current_annotation(None)

    def _save(self):
        if not self._canvas.is_loaded:
            QMessageBox.warning(self, "No data", "Load a datacube first.")
            return

        hdr_path = self._current_hdr_path
        if not hdr_path:
            return

        stem = os.path.splitext(hdr_path)[0]
        for ext in [".bsq.hdr", ".hdr", ".HDR"]:
            if stem.lower().endswith(ext.lower()):
                stem = stem[:len(stem) - len(ext)]
                break

        default_path = f"{stem}.json"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotations", default_path,
            "COCO JSON (*.json)")
        if not path:
            return

        out_dir = os.path.dirname(os.path.abspath(path))
        fname = os.path.basename(hdr_path)

        # ── Sync all annotations from layer pixels before saving ──
        for ann in self._annot_registry.all():
            self._sync_annotation_from_layer(ann["id"])

        categories = self._cat_registry.as_list()
        coco_categories = [
            {"id": int(cid), "name": str(name), "supercategory": ""}
            for cid, name, _ in categories
        ]

        coco_annotations = []
        for ann in self._annot_registry.all():
            seg  = ann.get("segmentation", [])
            bbox = ann.get("bbox", [])
            area = ann.get("area", 0)
            coco_seg = seg if (seg and isinstance(seg[0], list)) \
                else ([seg] if seg else [])
            coco_annotations.append({
                "id":           ann["id"],
                "image_id":     1,
                "category_id":  ann["category_id"],
                "segmentation": coco_seg,
                "area":         area,
                "bbox":         bbox,
                "iscrowd":      0,
            })

        coco = {
            "info": {
                "description": "HSI annotation (COCO)",
                "version":     "1.0",
                "year":        2026,
                "datacube":    fname,
            },
            "licenses":    [],
            "images": [{
                "id":        1,
                "file_name": fname,
                "width":     self._canvas._layer_w,
                "height":    self._canvas._layer_h,
            }],
            "annotations": coco_annotations,
            "categories":  coco_categories,
        }

        ok = False
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(coco, fh, ensure_ascii=False, indent=2)
            ok = True
        except Exception:
            log.exception("Failed to save COCO JSON")

        # ── Export RGB ──
        rgb_out = os.path.splitext(path)[0] + "_rgb.png"
        try:
            rgb_img = self._cube_rgb_cache.get(hdr_path)
            if rgb_img is None:
                log.info("[save] RGB cache miss, rendering...")
                rgb_img = self._canvas.render_preview(
                    self._preview_low_cut, self._preview_high_cut)

            if rgb_img is not None:
                saved = rgb_img.save(rgb_out)
                log.info("[save] RGB → %s (saved=%s, %dx%d)",
                         rgb_out, saved,
                         rgb_img.width(), rgb_img.height())
            else:
                log.warning("[save] RGB render returned None")
        except Exception:
            log.exception("[save] RGB export failed")

    # def _save(self):
    #     if not self._canvas.is_loaded:
    #         QMessageBox.warning(self, "No data", "Load a datacube first.")
    #         return

    #     workspace_dir = self._workspace_panel._current_dir or ""
    #     if workspace_dir:
    #         ws_name = os.path.basename(workspace_dir.rstrip(os.sep))
    #         default_path = os.path.join(workspace_dir, f"{ws_name}.json")
    #     else:
    #         default_path = "annotations.json"

    #     path, _ = QFileDialog.getSaveFileName(
    #         self, "Save All Annotations", default_path,
    #         "COCO JSON (*.json)")
    #     if not path:
    #         return

    #     # Save current cube state to cache first
    #     if self._current_hdr_path:
    #         self._save_cube_state()

    #     out_dir = os.path.dirname(os.path.abspath(path))

    #     categories = self._cat_registry.as_list()
    #     coco_categories = [
    #         {"id": int(cid), "name": str(name), "supercategory": ""}
    #         for cid, name, _ in categories
    #     ]

    #     coco_images = []
    #     coco_annotations = []
    #     image_id = 1
    #     id_to_hdr = {}

    #     # ── Get ALL hdr files from workspace ──
    #     if workspace_dir:
    #         all_hdr_files = list_hdr_files(workspace_dir)
    #     else:
    #         all_hdr_files = []
    #         if self._current_hdr_path:
    #             all_hdr_files.append(self._current_hdr_path)

    #     for hdr_path in all_hdr_files:
    #         fname = os.path.basename(hdr_path)
    #         cached = self._cube_ann_cache.get(hdr_path)

    #         # Width/height from cache or zero
    #         cube_w = cached.get("width", 0) if cached else 0
    #         cube_h = cached.get("height", 0) if cached else 0

    #         id_to_hdr[image_id] = hdr_path

    #         coco_images.append({
    #             "id":        image_id,
    #             "file_name": fname,
    #             "width":     cube_w,
    #             "height":    cube_h,
    #         })

    #         # Annotations from cache (empty if never visited)
    #         if cached:
    #             reg_state = cached.get("registry", {})
    #             anns = reg_state.get("annotations", [])
    #             for ann in anns:
    #                 seg  = ann.get("segmentation", [])
    #                 bbox = ann.get("bbox", [])
    #                 area = ann.get("area", 0)
    #                 coco_seg = seg if (
    #                     seg and isinstance(seg[0], list)) \
    #                     else ([seg] if seg else [])
    #                 coco_annotations.append({
    #                     "id":           ann["id"],
    #                     "image_id":     image_id,
    #                     "category_id":  ann["category_id"],
    #                     "segmentation": coco_seg,
    #                     "area":         area,
    #                     "bbox":         bbox,
    #                     "iscrowd":      0,
    #                 })

    #         image_id += 1

    #     coco = {
    #         "info": {
    #             "description": "HSI workspace annotations (COCO)",
    #             "version":     "1.0",
    #             "year":        2026,
    #         },
    #         "licenses":    [],
    #         "images":      coco_images,
    #         "annotations": coco_annotations,
    #         "categories":  coco_categories,
    #     }

    #     ok = False
    #     try:
    #         with open(path, "w", encoding="utf-8") as fh:
    #             json.dump(coco, fh, ensure_ascii=False, indent=2)
    #         ok = True
    #     except Exception:
    #         log.exception("Failed to save COCO JSON")

    #     # ── Export RGB for ALL cubes ──
    #     rgb_saved = 0
    #     rgb_skipped = 0
    #     rgb_failed = 0

    #     for img_entry in coco_images:
    #         img_id   = img_entry["id"]
    #         hdr_path = id_to_hdr.get(img_id, "")
    #         fname    = img_entry["file_name"]

    #         stem = fname
    #         for ext in [".bsq.hdr", ".hdr", ".HDR"]:
    #             if stem.lower().endswith(ext.lower()):
    #                 stem = stem[:len(stem) - len(ext)]
    #                 break

    #         rgb_out = os.path.join(out_dir, f"{stem}_rgb.png")

    #         if os.path.isfile(rgb_out):
    #             rgb_skipped += 1
    #             continue

    #         try:
    #             # Get from cache or load fresh
    #             rgb_img = self._cube_rgb_cache.get(hdr_path)
    #             if rgb_img is None:
    #                 from ..data import load_datacube_preview
    #                 _, rgb_img, _ = load_datacube_preview(
    #                     hdr_path,
    #                     low_cut=self._preview_low_cut,
    #                     high_cut=self._preview_high_cut)
    #                 if rgb_img:
    #                     self._cube_rgb_cache[hdr_path] = rgb_img

    #             if rgb_img and rgb_img.save(rgb_out):
    #                 rgb_saved += 1
    #                 log.info("RGB exported: %s", rgb_out)
    #             else:
    #                 rgb_failed += 1
    #                 log.warning("RGB export failed: %s", hdr_path)
    #         except Exception:
    #             rgb_failed += 1
    #             log.exception("RGB export error: %s", hdr_path)

    #     # ── Status ──
    #     msg = []
    #     if ok:
    #         msg.append(f"Saved {os.path.basename(path)}")
    #     else:
    #         msg.append("Save failed")
    #     msg.append(f"{len(coco_images)} cubes")
    #     msg.append(f"{len(coco_annotations)} anns")
    #     if rgb_saved:
    #         msg.append(f"{rgb_saved} RGB exported")
    #     if rgb_skipped:
    #         msg.append(f"{rgb_skipped} RGB skipped")
    #     if rgb_failed:
    #         msg.append(f"{rgb_failed} RGB failed")
    #     self.statusBar().showMessage(" | ".join(msg), 8000)

    # ------------------------------------------------------------------
    # Create new annotation
    # ------------------------------------------------------------------

    def _create_new_annotation(self):
        cat_id = self._category_panel.active_category_id()
        if cat_id is None:
            QMessageBox.warning(
                self, "No category",
                "Create a category first.")
            return
        ann_id = self._annot_registry.add(cat_id)
        log.info("Created annotation #%d  category=%d (%s)",
                 ann_id, cat_id, self._cat_registry.name(cat_id))

    # ------------------------------------------------------------------
    # Active annotation → canvas
    # ------------------------------------------------------------------

    def _on_active_annotation_changed(self, ann_id):
        if ann_id is None:
            self._canvas.set_current_annotation(None)
            self.statusBar().showMessage(
                "Select an annotation to start drawing", 0)
            return
        ann = self._annot_registry.get(ann_id)
        if ann is None:
            return
        cat_id    = ann["category_id"]
        cat_name  = self._cat_registry.name(cat_id)
        cat_color = self._cat_registry.qcolor(cat_id, alpha=220)
        self._canvas.set_current_annotation(
            ann_id, cat_id, cat_name, cat_color)
        self.statusBar().showMessage(
            "Drawing #{} → {}  |  L=connect  |  C=circle".format(
                ann_id, cat_name), 0)


    def _update_eraser_clip(self, ann_id):
        """Update canvas eraser clip polygons from annotation's segmentation."""
        ann = self._annot_registry.get(ann_id)
        if ann is None:
            self._canvas._eraser_clip_polygons = []
            return
        seg = ann.get("segmentation", [])
        from ..canvas import CanvasItem
        self._canvas._eraser_clip_polygons = \
            CanvasItem._normalize_segmentation(seg)
    # ------------------------------------------------------------------
    # Shape completed → MERGE into annotation
    # ------------------------------------------------------------------

    @pyqtSlot(int, object, object, int)
    @pyqtSlot(int, object, object, int)
    def _on_shape_completed(self, ann_id, bbox, segmentation, area):
        if ann_id is None:
            return
        ann = self._annot_registry.get(ann_id)
        if ann is None:
            log.warning("Shape for unknown annotation %d", ann_id)
            return

        existing_bbox = ann.get("bbox", [])
        if existing_bbox and len(existing_bbox) >= 4:
            ex0, ey0 = existing_bbox[0], existing_bbox[1]
            ew, eh   = existing_bbox[2], existing_bbox[3]
            nx0, ny0 = bbox[0], bbox[1]
            nw, nh   = bbox[2], bbox[3]
            merged_bbox = [
                min(ex0, nx0), min(ey0, ny0),
                max(ex0 + ew, nx0 + nw) - min(ex0, nx0),
                max(ey0 + eh, ny0 + nh) - min(ey0, ny0),
            ]
        else:
            merged_bbox = list(bbox)

        existing_seg = ann.get("segmentation", [])
        if existing_seg and isinstance(existing_seg[0], list):
            merged_seg = existing_seg + [list(segmentation)]
        elif existing_seg:
            merged_seg = [existing_seg, list(segmentation)]
        else:
            merged_seg = [list(segmentation)]

        merged_area = (ann.get("area", 0) or 0) + area

        self._annot_registry.update(
            ann_id,
            bbox=merged_bbox,
            segmentation=merged_seg,
            area=merged_area,
        )
        # Update eraser clip to include new polygon
        self._update_eraser_clip(ann_id)
        self._update_bbox_overlays()
        log.debug("Annotation #%d — %d seg(s), area=%d",
                  ann_id, len(merged_seg), merged_area)
        
    def _on_shape_closed(self):
        """After eraser: sync annotation from layer pixels."""
        if self._canvas._tool != "eraser":
            return
        ann_id = self._annotation_panel.active_annotation_id()
        if ann_id is None:
            return
        self._sync_annotation_from_layer(ann_id)
        self._refresh_pg()
        self._update_bbox_overlays()
        self._compute_category_spectra()
    # ------------------------------------------------------------------
    # Helpers for eraser recompute
    # ------------------------------------------------------------------

    def _clear_annotation_from_mask(self, polygons, color_rgb, w, h):
        """Remove annotation pixels from mask using polygon clip."""
        mask_img = self._canvas._mask
        # Build clip region from polygons (fill + edge)
        clip_img = QImage(w, h, QImage.Format_Grayscale8)
        clip_img.fill(0)
        cp = QPainter(clip_img)
        pen = QPen(QColor(255), self._canvas._pen_width,
                   Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        cp.setPen(pen)
        cp.setBrush(QColor(255))
        for poly in polygons:
            from ..canvas import CanvasItem
            pts = CanvasItem._segmentation_to_points(poly)
            if len(pts) >= 3:
                cp.drawPolygon(QPolygonF(pts))
        cp.end()
        ptr_c = clip_img.bits()
        ptr_c.setsize(h * w)
        clip = np.frombuffer(ptr_c, np.uint8).reshape((h, w)) > 0

        # Clear pixels in clip region that match category color
        rgba = mask_img.convertToFormat(QImage.Format_RGBA8888)
        ptr = rgba.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4)).copy()

        from ..data import _match_color
        color_match = _match_color(arr, QColor(*color_rgb, alpha=255))
        clear_mask = clip & color_match
        arr[clear_mask] = 0

        self._canvas._mask = QImage(
            arr.data, w, h, w * 4, QImage.Format_RGBA8888
        ).copy().convertToFormat(QImage.Format_ARGB32)

    def _detect_contour_polygons(self, binary_mask):
        """Detect ALL contours (including holes) from a binary mask.
        Uses RETR_TREE to capture nested contours (donut shapes).
        Returns list of flat polygon lists [[x1,y1,x2,y2,...], ...]
        """
        try:
            import cv2
            contours, hierarchy = cv2.findContours(
                binary_mask.astype(np.uint8),
                cv2.RETR_TREE,              # ← เปลี่ยนจาก RETR_EXTERNAL
                cv2.CHAIN_APPROX_SIMPLE,
            )
            polygons = []
            for contour in contours:
                c = contour.squeeze(axis=1)
                if len(c) < 3:
                    continue
                flat = c.flatten().tolist()
                if len(flat) >= 6:
                    polygons.append(flat)
            return polygons
        except ImportError:
            pass

        # Fallback: bounding box polygon per connected component
        visited = np.zeros_like(binary_mask, dtype=bool)
        h, w = binary_mask.shape
        polygons = []
        for y in range(h):
            for x in range(w):
                if not binary_mask[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                pixels = []
                while stack:
                    cy, cx = stack.pop()
                    pixels.append((cx, cy))
                    for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                        if (0 <= ny < h and 0 <= nx < w
                                and not visited[ny, nx]
                                and binary_mask[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if len(pixels) < 3:
                    continue
                xs = [p[0] for p in pixels]
                ys = [p[1] for p in pixels]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                polygons.append([
                    float(x0), float(y0),
                    float(x1), float(y0),
                    float(x1), float(y1),
                    float(x0), float(y1),
                ])
        return polygons

    def _recompute_annotation_from_mask(self, ann_id):
        """Recompute bbox and area for annotation from remaining mask pixels."""
        ann = self._annot_registry.get(ann_id)
        if ann is None:
            return

        cat_id = ann["category_id"]
        color  = self._cat_registry.color(cat_id)

        mask_img = self._canvas.get_mask()
        w, h = mask_img.width(), mask_img.height()

        rgba = mask_img.convertToFormat(QImage.Format_RGBA8888)
        ptr = rgba.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))

        match = (
            (np.abs(arr[:, :, 0].astype(int) - color[0]) <= 5)
            & (np.abs(arr[:, :, 1].astype(int) - color[1]) <= 5)
            & (np.abs(arr[:, :, 2].astype(int) - color[2]) <= 5)
            & (arr[:, :, 3] > 0)
        )

        old_bbox = ann.get("bbox", [])
        if old_bbox and len(old_bbox) >= 4:
            bx0 = max(0, int(old_bbox[0]) - 2)
            by0 = max(0, int(old_bbox[1]) - 2)
            bx1 = min(w, int(old_bbox[0]) + int(old_bbox[2]) + 2)
            by1 = min(h, int(old_bbox[1]) + int(old_bbox[3]) + 2)
            region = np.zeros((h, w), dtype=bool)
            region[by0:by1, bx0:bx1] = True
            match = match & region

        ys, xs = np.where(match)
        n = len(ys)

        if n == 0:
            # All pixels erased — clear bbox, keep annotation alive
            self._annot_registry.update(
                ann_id, bbox=[0, 0, 0, 0], area=0)
            log.debug("Annotation #%d — all pixels erased, bbox cleared",
                      ann_id)
        else:
            nx0, ny0 = int(xs.min()), int(ys.min())
            nx1, ny1 = int(xs.max()), int(ys.max())
            new_bbox = [nx0, ny0, nx1 - nx0 + 1, ny1 - ny0 + 1]
            self._annot_registry.update(ann_id, bbox=new_bbox, area=n)
            log.debug("Annotation #%d — %d px remain, bbox=%s",
                      ann_id, n, new_bbox)
    # ------------------------------------------------------------------
    # Annotation panel → canvas operations
    # ------------------------------------------------------------------

    def _on_annotation_category_reassigned(self, ann_id, old_cat_id,
                                           new_cat_id):
        old_color = self._cat_registry.color(old_cat_id)
        new_color = self._cat_registry.color(new_cat_id)
        self._canvas.recolor_annotation_layer(
            ann_id, old_color, new_color)

        if ann_id == self._annotation_panel.active_annotation_id():
            cat_name  = self._cat_registry.name(new_cat_id)
            cat_color = self._cat_registry.qcolor(new_cat_id, alpha=220)
            self._canvas.set_current_annotation(
                ann_id, new_cat_id, cat_name, cat_color)

        self._refresh_pg()
        self._update_bbox_overlays()
        self._compute_category_spectra()

    def _on_annotation_delete_requested(self, ann_id):
        self._canvas.remove_layer(ann_id)
        self._annot_registry.remove(ann_id)
        self._refresh_pg()
        self._update_bbox_overlays()
        self._compute_category_spectra()

    def _on_annotation_visibility_changed(self, ann_id):
        visible = self._annot_registry.is_visible(ann_id)
        self._canvas.set_layer_visible(ann_id, visible)
        self._refresh_pg()
        self._update_bbox_overlays()

    def _rebuild_visible_mask(self):
        self._canvas._composite()

    # ------------------------------------------------------------------
    # Category registry → canvas sync
    # ------------------------------------------------------------------

    def _on_registry_category_changed(self, category_id):
        vis = self._cat_registry.is_visible(category_id)

        # Batch update all annotation layers of this category
        anns = self._annot_registry.by_category(category_id)
        if anns:
            vis_map = {ann["id"]: vis for ann in anns}
            self._canvas.set_layers_visible_batch(vis_map)

        # Update pen if active annotation uses this category
        active_id = self._annotation_panel.active_annotation_id()
        if active_id is not None:
            ann = self._annot_registry.get(active_id)
            if ann and ann["category_id"] == category_id:
                name = self._cat_registry.name(category_id)
                qc = self._cat_registry.qcolor(category_id, alpha=220)
                self._canvas.set_current_annotation(
                    active_id, category_id, name, qc)

        self._update_bbox_overlays()

    def _on_category_about_to_be_removed(self, category_id, old_color):
        if self._canvas is None or not self._canvas.is_loaded:
            return
        # Remove all annotation layers of this category
        for ann in self._annot_registry.by_category(category_id):
            self._canvas.remove_layer(ann["id"])

    def _on_registry_category_removed(self, category_id):
        log.info("Category removed: id=%d", category_id)
        self._update_bbox_overlays()

    def _on_registry_category_color_changed(self, category_id,
                                            old_color, new_color):
        if self._canvas is None or not self._canvas.is_loaded:
            return
        # Recolor all annotation layers of this category
        for ann in self._annot_registry.by_category(category_id):
            self._canvas.recolor_annotation_layer(
                ann["id"], old_color, new_color)
        self._refresh_pg()
        self._update_bbox_overlays()
        self._compute_category_spectra()

    # ------------------------------------------------------------------
    # Canvas / pg events
    # ------------------------------------------------------------------

    def _on_loaded(self, ncols, nrows, nbands):
        log.info("Datacube loaded — %dx%d  bands=%d",
                 ncols, nrows, nbands)
        preview = self._format_preview_info()
        self.statusBar().showMessage(
            "Loaded  |  {}×{} px  |  {} bands  |  {}"
            "  |  Ctrl+Wheel=Zoom  |  Ctrl+S=save".format(
                ncols, nrows, nbands, preview), 0)
        for action in self._tool_actions.values():
            action.setEnabled(True)
        self._contrast_action.setEnabled(True)
        # Cache RGB preview for this cube
        rgb_img = self._canvas.render_preview(
            self._preview_low_cut, self._preview_high_cut)
        if rgb_img:
            self._cube_rgb_cache[self._current_hdr_path] = rgb_img

    def _refresh_pg(self):
        pass  # PgPanel shows spectrum only, no image view

    def _update_bbox_overlays(self):
        for it in list(self._bbox_items):
            try:
                self._scene.removeItem(it)
            except Exception:
                pass
        self._bbox_items.clear()

        if not getattr(self._canvas, "is_loaded", False):
            return

        for ann in self._annot_registry.all():
            if not ann.get("visible", True):
                continue
            cat_id = ann.get("category_id")
            if cat_id is None:
                continue
            try:
                if not self._cat_registry.is_visible(cat_id):
                    continue
            except Exception:
                pass

            bbox = ann.get("bbox", [])
            if len(bbox) < 4:
                continue
            try:
                x0, y0, w, h = (float(bbox[0]), float(bbox[1]),
                                 float(bbox[2]), float(bbox[3]))
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue

            cat_name = self._cat_registry.name(cat_id)
            ann_id   = ann["id"]
            area     = ann.get("area", 0) or 0
            seg_data = ann.get("segmentation", [])
            n_seg = len(seg_data) if (seg_data and isinstance(seg_data[0], list)) else (1 if seg_data else 0)

            pen = QPen(self._cat_registry.qcolor(cat_id, alpha=200))
            pen.setWidth(2)
            rect = QGraphicsRectItem(x0, y0, w, h)
            rect.setPen(pen)
            rect.setBrush(QBrush())
            rect.setZValue(100)
            self._scene.addItem(rect)
            self._bbox_items.append(rect)

            try:
                label_str = "#{} ({})  {} seg  {} px".format(
                    ann_id, cat_name, n_seg, area)
                txt = QGraphicsSimpleTextItem(label_str)
                font = QFont()
                font.setPointSize(8)
                font.setBold(True)
                txt.setFont(font)
                txt.setBrush(QBrush(QColor(255, 255, 255)))

                br = txt.boundingRect()
                pad_x, pad_y = 4, 2
                bg_w = br.width()  + pad_x * 2
                bg_h = br.height() + pad_y * 2
                tag_y = max(0, y0 - bg_h)

                bg = QGraphicsRectItem(x0, tag_y, bg_w, bg_h)
                bg.setBrush(QBrush(
                    self._cat_registry.qcolor(cat_id, alpha=210)))
                bg.setPen(QPen(Qt.NoPen))
                bg.setZValue(100)
                self._scene.addItem(bg)
                self._bbox_items.append(bg)

                txt.setPos(QPointF(x0 + pad_x, tag_y + pad_y))
                txt.setZValue(101)
                self._scene.addItem(txt)
                self._bbox_items.append(txt)
            except Exception:
                log.exception("Failed to add bbox label for #%d", ann_id)

    def _compute_category_spectra(self):
        if self._canvas.datacube is None or self._canvas.is_drawing:
            return
        if self._spectra_running:
            self._spectra_pending = True
            return
        self._pg_panel.clear_cursor_spectrum()
        self._spectra_running = True
        self._spectra_pending = False
        self._pg_panel.set_spectrum_status("Computing spectra...")
        self.category_spectra_request.emit(
            self._canvas.datacube,
            self._canvas.get_mask(),
            self._cat_registry.as_list(),
        )

    def _on_category_spectra_ready(self, data):
        log.debug("Spectra ready — %d class(es)", len(data))
        self._spectra_running = False
        self._pg_panel.update_class_spectra(data)
        if self._spectra_pending:
            self._compute_category_spectra()

    def _on_category_spectra_error(self, message):
        log.error("Spectra error: %s", message)
        self._spectra_running = False
        self._spectra_pending = False
        try:
            self._pg_panel.set_spectrum_status("Spectrum compute failed")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _format_preview_info(self):
        info   = self._canvas.preview_info or {}
        low    = info.get("low_cut",  self._preview_low_cut)
        high   = info.get("high_cut", self._preview_high_cut)
        actual = info.get("actual_wavelengths")
        if actual:
            wl_text = "RGB~{:.0f}/{:.0f}/{:.0f} nm".format(*actual)
        else:
            bands = info.get("band_indices")
            wl_text = (
                "RGB bands={}/{}/{}".format(*bands)
                if bands else "RGB fallback")
        return "{}  cut {:.1f}-{:.1f}%".format(wl_text, low, high)

    def _default_gt_filename(self):
        dc  = self._canvas.datacube
        src = getattr(dc, "filename", "") if dc else ""
        name = os.path.basename(src)
        if not name:
            return "gt_mask"
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".hdr":
            name = stem
        for suf in (".bip", ".bil", ".bsq"):
            if name.lower().endswith(suf):
                name = name[: -len(suf)]
                break
        return "{}".format(name or "gt_mask")

    def closeEvent(self, event):
        try:
            self.category_spectra_request.disconnect(
                self._spectrum_worker.process)
        except Exception:
            pass
        try:
            self._spectrum_worker.finished.disconnect()
            self._spectrum_worker.error.disconnect()
        except Exception:
            pass
        self._spectrum_thread.quit()
        if not self._spectrum_thread.wait(3000):
            self._spectrum_thread.terminate()
            self._spectrum_thread.wait(1000)
        from PyQt5.QtCore import QThreadPool
        QThreadPool.globalInstance().waitForDone(3000)
        super().closeEvent(event)

    def _sync_annotation_from_layer(self, ann_id):
        """Sync registry segmentation + bbox from actual layer pixels."""
        layer_np = self._canvas.layer_to_numpy(ann_id)
        if layer_np is None:
            return

        remaining = layer_np[:, :, 3] > 0
        remain_count = int(np.count_nonzero(remaining))

        if remain_count == 0:
            self._annot_registry.update(
                ann_id, bbox=[0, 0, 0, 0],
                segmentation=[], area=0)
            log.info("[sync] #%d — empty layer", ann_id)
            return

        new_polygons = self._detect_contour_polygons(
            remaining.astype(np.uint8))
        log.info("[sync] #%d — detected %d contour(s) from %d px",
                 ann_id, len(new_polygons), remain_count)

        if not new_polygons:
            fys, fxs = np.where(remaining)
            if len(fys) > 0:
                nx0, ny0 = int(fxs.min()), int(fys.min())
                nx1, ny1 = int(fxs.max()), int(fys.max())
                new_polygons = [[
                    float(nx0), float(ny0),
                    float(nx1), float(ny0),
                    float(nx1), float(ny1),
                    float(nx0), float(ny1)]]

        all_xs, all_ys = [], []
        for poly in new_polygons:
            pts = CanvasItem._segmentation_to_points(poly)
            for p in pts:
                all_xs.append(p.x())
                all_ys.append(p.y())

        if all_xs:
            new_bbox = [min(all_xs), min(all_ys),
                        max(all_xs) - min(all_xs),
                        max(all_ys) - min(all_ys)]
        else:
            new_bbox = [0, 0, 0, 0]

        self._annot_registry.update(
            ann_id,
            bbox=new_bbox,
            segmentation=new_polygons,
            area=remain_count,
        )
        log.info("[sync] #%d → %d poly(s), %d px, bbox=%s",
                 ann_id, len(new_polygons), remain_count, new_bbox)