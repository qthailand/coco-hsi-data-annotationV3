# hsi_annotation/canvas.py
import logging
import time
import math
from collections import deque

import numpy as np
from PyQt5.QtCore import QObject, QPointF, QRunnable, QThreadPool, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QGraphicsPixmapItem

from .data import (
    DEFAULT_HIGH_CUT,
    DEFAULT_LOW_CUT,
    _match_color,
    build_rgb_preview,
    load_datacube_preview,
)

log = logging.getLogger(__name__)


class _SpecSignals(QObject):
    ready = pyqtSignal(int, int, object, int)


class _SpectrumRunnable(QRunnable):
    def __init__(self, datacube, x, y, seq, seq_ref):
        super().__init__()
        self.signals = _SpecSignals()
        self._datacube = datacube
        self._x = x
        self._y = y
        self._seq = seq
        self._seq_ref = seq_ref

    def run(self):
        if self._seq != self._seq_ref[0]:
            return
        try:
            spec = np.array(
                self._datacube[self._y, self._x, :],
                dtype=np.float32).flatten()
            self.signals.ready.emit(self._x, self._y, spec, self._seq)
        except Exception:
            log.exception("SpectrumRunnable: failed (%d,%d)",
                          self._x, self._y)


class CanvasSignals(QObject):
    updated = pyqtSignal()
    shape_closed = pyqtSignal()
    shape_completed = pyqtSignal(int, object, object, int)
    spectrum_ready = pyqtSignal(int, int, object)
    loaded = pyqtSignal(int, int, int)
    move_completed = pyqtSignal(int, float, float)
    move_hit_annotation = pyqtSignal(int)  # ann_id to select


class BgItem(QGraphicsPixmapItem):
    def __init__(self):
        super().__init__()
        self.setZValue(0)
        self.setAcceptHoverEvents(False)
    def mousePressEvent(self, event):   event.ignore()
    def mouseMoveEvent(self, event):    event.ignore()
    def mouseReleaseEvent(self, event): event.ignore()


class CanvasItem(QGraphicsPixmapItem):
    def __init__(self):
        super().__init__()
        self.signals = CanvasSignals()
        self._datacube = None
        self._is_loaded = False
        self._drawing = False
        self._last_pos = QPointF()
        self._connect_start = None
        self._connect_last = None
        self._connect_points = []
        self._last_spectrum_emit = 0.0
        self._spectrum_interval_s = 0.06
        self._spectrum_seq = 0
        self._spectrum_seq_ref = [0]

        self._current_annotation_id = None
        self._current_category_id = None
        self._current_category_name = ""
        self._pen_color = QColor(0, 0, 0, 0)
        self._pen_width = 4
        self._tool = "connect"
        self._preview_low_cut = DEFAULT_LOW_CUT
        self._preview_high_cut = DEFAULT_HIGH_CUT
        self._preview_info = None
        self._circle_center = None
        self._circle_radius = 0
        self._circle_base_mask = None

        self._moving_ann_id = None
        self._moving_start  = None
        self._total_move_dx = 0.0
        self._total_move_dy = 0.0
        self._rgb_array = None       # (h, w, 3) uint8 for wand
        self._wand_tolerance = 30

        # ── Layer system ──
        self._layers = {}         # {ann_id: QImage ARGB32}
        self._layer_visible = {}  # {ann_id: bool}
        self._layer_w = 800
        self._layer_h = 600

        # Display composite
        self._mask = QImage(800, 600, QImage.Format_ARGB32)
        self._mask.fill(0)
        self.setPixmap(QPixmap.fromImage(self._mask))
        self.setZValue(1)
        self.setOpacity(0.75)
        self.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)
        self.setAcceptHoverEvents(True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def datacube(self):           return self._datacube
    @property
    def is_loaded(self):          return self._is_loaded
    @property
    def is_drawing(self):         return self._drawing
    @property
    def preview_info(self):       return self._preview_info
    @property
    def preview_low_cut(self):    return self._preview_low_cut
    @property
    def preview_high_cut(self):   return self._preview_high_cut
    @property
    def has_active_annotation(self):
        return self._current_annotation_id is not None
    @property
    def current_annotation_id(self):
        return self._current_annotation_id

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def _get_or_create_layer(self, ann_id):
        if ann_id not in self._layers:
            img = QImage(self._layer_w, self._layer_h,
                         QImage.Format_ARGB32)
            img.fill(0)
            self._layers[ann_id] = img
            self._layer_visible[ann_id] = True
        return self._layers[ann_id]

    def remove_layer(self, ann_id):
        self._layers.pop(ann_id, None)
        self._layer_visible.pop(ann_id, None)
        self._composite()

    def set_layer_visible(self, ann_id, visible):
        self._layer_visible[ann_id] = bool(visible)
        self._composite()

    def move_layer(self, ann_id, dx, dy):
        """Translate layer pixels by (dx, dy)."""
        layer = self._layers.get(ann_id)
        if layer is None:
            return
        w, h = self._layer_w, self._layer_h
        new_layer = QImage(w, h, QImage.Format_ARGB32)
        new_layer.fill(0)
        painter = QPainter(new_layer)
        painter.drawImage(int(dx), int(dy), layer)
        painter.end()
        self._layers[ann_id] = new_layer
        self._composite()

    def _hit_test_layers(self, pos):
        """Find top-most annotation with non-transparent pixel at pos."""
        x, y = int(pos.x()), int(pos.y())
        if not (0 <= x < self._layer_w and 0 <= y < self._layer_h):
            return None
        # Check in reverse order (top layers first)
        for ann_id in reversed(sorted(self._layers.keys())):
            if not self._layer_visible.get(ann_id, True):
                continue
            layer = self._layers[ann_id]
            pixel = layer.pixel(x, y)
            alpha = (pixel >> 24) & 0xFF
            if alpha > 0:
                return ann_id
        return None

    def layer_to_numpy(self, ann_id):
        """Read annotation layer as RGBA numpy array."""
        layer = self._layers.get(ann_id)
        if layer is None:
            return None
        w, h = self._layer_w, self._layer_h
        rgba = layer.convertToFormat(QImage.Format_RGBA8888)
        ptr = rgba.bits()
        ptr.setsize(h * w * 4)
        return np.frombuffer(ptr, np.uint8).reshape(
            (h, w, 4)).copy()

    def _composite(self):
        """Merge all visible layers into display mask (last-write-wins)."""
        w, h = self._layer_w, self._layer_h
        composite = np.zeros((h, w, 4), dtype=np.uint8)

        for ann_id in sorted(self._layers.keys()):
            if not self._layer_visible.get(ann_id, True):
                continue
            layer = self._layers[ann_id]
            rgba = layer.convertToFormat(QImage.Format_RGBA8888)
            ptr = rgba.bits()
            ptr.setsize(h * w * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
            has_content = arr[:, :, 3] > 0
            composite[has_content] = arr[has_content]

        self._mask = QImage(
            composite.data, w, h, w * 4, QImage.Format_RGBA8888
        ).copy().convertToFormat(QImage.Format_ARGB32)
        self.setPixmap(QPixmap.fromImage(self._mask))

    # ------------------------------------------------------------------
    # Init / setup
    # ------------------------------------------------------------------

    def set_tool(self, tool_name):
        log.debug("Tool changed: %s", tool_name)
        # Clear eraser cursor when switching away
        if self._tool == "eraser" and tool_name != "eraser":
            self._clear_eraser_cursor()
        self._tool = tool_name
        self._drawing = False
        if tool_name != "connect":
            self._connect_start = None
            self._connect_last = None
            self._connect_points = []

    def set_pen_color(self, color):
        self._pen_color = color

    def set_pen_width(self, width):
        self._pen_width = width

    def set_preview_cuts(self, low_cut, high_cut):
        self._preview_low_cut = float(low_cut)
        self._preview_high_cut = float(high_cut)

    def get_mask(self):
        return self._mask

    def set_current_annotation(self, ann_id, cat_id=None, name="",
                               color=None):
        self._current_annotation_id = ann_id
        self._current_category_id = cat_id
        self._current_category_name = name or ""
        if ann_id is None:
            log.info("No active annotation — drawing disabled")
            self._pen_color = QColor(0, 0, 0, 0)
        else:
            self._get_or_create_layer(ann_id)
            self._pen_color = (QColor(color) if color
                               else QColor(128, 128, 128))
            self._pen_color.setAlpha(220)

    def clear_mask(self):
        self._layers.clear()
        self._layer_visible.clear()
        self._mask.fill(0)
        self.setPixmap(QPixmap.fromImage(self._mask))
        self.signals.updated.emit()
        self.signals.shape_closed.emit()

    # ------------------------------------------------------------------
    # Datacube / preview
    # ------------------------------------------------------------------

    def load_datacube(self, path):
        log.info("CanvasItem.load_datacube: %s", path)
        self._datacube, rgb_img, self._preview_info = \
            load_datacube_preview(
                path,
                low_cut=self._preview_low_cut,
                high_cut=self._preview_high_cut)
        self._layer_w = rgb_img.width()
        self._layer_h = rgb_img.height()
        self._layers.clear()
        self._layer_visible.clear()
        self._mask = QImage(self._layer_w, self._layer_h,
                            QImage.Format_ARGB32)
        self._mask.fill(0)
        self.setPixmap(QPixmap.fromImage(self._mask))
        self._is_loaded = True
        log.info("Datacube loaded — %dx%d  %d bands",
                 self._layer_w, self._layer_h,
                 self._datacube.nbands)
        self.signals.loaded.emit(
            self._layer_w, self._layer_h, self._datacube.nbands)
        return rgb_img

    def render_preview(self, low_cut=None, high_cut=None):
        if self._datacube is None:
            return None
        if low_cut is not None:
            self._preview_low_cut = float(low_cut)
        if high_cut is not None:
            self._preview_high_cut = float(high_cut)
        rgb, self._preview_info = build_rgb_preview(
            self._datacube,
            low_cut=self._preview_low_cut,
            high_cut=self._preview_high_cut)
        rgb8 = np.ascontiguousarray(
            (rgb * 255).clip(0, 255).astype(np.uint8))
        height, width, _ = rgb8.shape
        return QImage(
            rgb8.data, width, height, width * 3,
            QImage.Format_RGB888
        ).convertToFormat(QImage.Format_ARGB32).copy()

    def _emit_spectrum(self, pos, force=False):
        if self._datacube is None:
            return
        now = time.perf_counter()
        if not force and (now - self._last_spectrum_emit) < self._spectrum_interval_s:
            return
        x, y = int(pos.x()), int(pos.y())
        if 0 <= x < self._datacube.ncols and 0 <= y < self._datacube.nrows:
            self._last_spectrum_emit = now
            self._spectrum_seq += 1
            seq = self._spectrum_seq
            self._spectrum_seq_ref[0] = seq
            worker = _SpectrumRunnable(
                self._datacube, x, y, seq,
                self._spectrum_seq_ref)
            worker.signals.ready.connect(
                self._on_spectrum_ready, Qt.QueuedConnection)
            QThreadPool.globalInstance().start(worker)

    def _on_spectrum_ready(self, x, y, arr, seq):
        if seq == self._spectrum_seq:
            self.signals.spectrum_ready.emit(x, y, arr)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if not self._is_loaded:
            return
        if event.button() == Qt.LeftButton:
            if self._tool == "probe":
                self._emit_spectrum(event.pos(), force=True)
                return
            if self._tool not in ("eraser", "move") and \
                    not self.has_active_annotation:
                log.warning("Drawing blocked: no active annotation")
                return
            if self._tool == "connect":
                self._connect_click(event.pos())
                self._emit_spectrum(event.pos(), force=True)
            elif self._tool == "circle":
                self._drawing = True
                self._circle_center = event.pos()
                self._circle_radius = 0
                self._circle_base_mask = QPixmap(self._mask).toImage()
                self._draw_circle_preview(0)
                self._emit_spectrum(event.pos(), force=True)
            elif self._tool == "eraser":
                self._drawing = True
                self._last_pos = event.pos()
                self._erase_at(event.pos())
                self._composite()
                self._update_eraser_cursor(event.pos())
                self._emit_spectrum(event.pos(), force=True)
            elif self._tool == "move":
                hit_id = self._hit_test_layers(event.pos())
                if hit_id is None:
                    return
                self._moving_ann_id = hit_id
                self._moving_start  = QPointF(event.pos())
                self._total_move_dx = 0.0
                self._total_move_dy = 0.0
                self._drawing = True
                if hit_id != self._current_annotation_id:
                    self.signals.move_hit_annotation.emit(hit_id)
            elif self._tool == "wand":
                polygons = self._wand_select(event.pos())
                if polygons:
                    color = QColor(self._pen_color)
                    color.setAlpha(220)
                    color_rgb = (color.red(), color.green(), color.blue())
                    self.paint_annotation_layer(
                        self._current_annotation_id,
                        polygons, color_rgb)
                    # Compute bbox from all polygons
                    all_xs, all_ys = [], []
                    for poly in polygons:
                        pts = self._segmentation_to_points(poly)
                        for p in pts:
                            all_xs.append(p.x())
                            all_ys.append(p.y())
                    if all_xs:
                        bbox = [min(all_xs), min(all_ys),
                                max(all_xs) - min(all_xs),
                                max(all_ys) - min(all_ys)]
                        # Count pixels
                        layer_np = self.layer_to_numpy(
                            self._current_annotation_id)
                        area = int(np.count_nonzero(
                            layer_np[:, :, 3] > 0)) \
                            if layer_np is not None else 0
                        self.signals.shape_completed.emit(
                            self._current_annotation_id,
                            bbox, polygons, area)
                    self.signals.shape_closed.emit()
                    self.signals.updated.emit()
        elif event.button() == Qt.RightButton \
                and self._tool == "connect":
            self._close_connect_path()

    def mouseMoveEvent(self, event):
        if not self._is_loaded:
            return
        if self._tool == "circle":
            if self._drawing and (event.buttons() & Qt.LeftButton):
                dx = event.pos().x() - self._circle_center.x()
                dy = event.pos().y() - self._circle_center.y()
                self._circle_radius = int(math.hypot(dx, dy))
                self._draw_circle_preview(self._circle_radius)
            return
        if self._tool in ("fill", "connect"):
            return
        if self._drawing and (event.buttons() & Qt.LeftButton):
            if self._tool == "eraser":
                self._erase_line(self._last_pos, event.pos())
                self._last_pos = event.pos()
                self._composite()
                self._update_eraser_cursor(event.pos())
            elif self._tool == "move" and self._moving_ann_id is not None:
                dx = event.pos().x() - self._moving_start.x()
                dy = event.pos().y() - self._moving_start.y()
                self.move_layer(self._moving_ann_id, dx, dy)
                self._moving_start = QPointF(event.pos())
                self._total_move_dx += dx
                self._total_move_dy += dy
            elif self._tool not in ("eraser", "move"):
                self._draw_line(self._last_pos, event.pos())
                self._last_pos = event.pos()
            self._emit_spectrum(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._tool == "eraser":
                self._drawing = False
                self._composite()
                self._update_eraser_cursor(event.pos())
                self.signals.updated.emit()
                self.signals.shape_closed.emit()
                self._emit_spectrum(event.pos(), force=True)
            elif self._tool == "move" and self._moving_ann_id is not None:
                ann_id = self._moving_ann_id
                dx = self._total_move_dx
                dy = self._total_move_dy
                self._moving_ann_id = None
                self._moving_start  = None
                self._total_move_dx = 0.0
                self._total_move_dy = 0.0
                if abs(dx) >= 0.5 or abs(dy) >= 0.5:
                    self.signals.move_completed.emit(ann_id, dx, dy)
            elif self._tool == "circle":
                if self._drawing and self._circle_radius > 0:
                    self._paint_circle(self._circle_center,
                                       self._circle_radius)
                    poly_pts = self._circle_to_polygon(
                        self._circle_center, self._circle_radius)
                    bbox = self._compute_polygon_bbox(poly_pts)
                    seg = self._flatten_points(poly_pts)
                    area = int(math.pi * self._circle_radius ** 2)
                    self.signals.shape_completed.emit(
                        self._current_annotation_id,
                        bbox, seg, area)
                self._drawing = False
                self._circle_center = None
                self._circle_radius = 0
                self._circle_base_mask = None
                self.signals.updated.emit()
                self.signals.shape_closed.emit()
                self._emit_spectrum(event.pos(), force=True)

    def mouseDoubleClickEvent(self, event):
        event.ignore()

    def hoverLeaveEvent(self, event):
        if self._tool == "eraser":
            self._clear_eraser_cursor()

    # ------------------------------------------------------------------
    # Drawing on active layer
    # ------------------------------------------------------------------

    def _paint_on_active_layer(self, painter_fn):
        """Paint on the active annotation's layer, then recomposite."""
        ann_id = self._current_annotation_id
        if ann_id is None:
            return
        layer = self._get_or_create_layer(ann_id)
        painter = QPainter(layer)
        if self._tool == "eraser":
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
        else:
            pen = QPen(self._pen_color, self._pen_width,
                       Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
        painter_fn(painter)
        painter.end()
        self._composite()

    def _draw_dot(self, pos):
        self._paint_on_active_layer(lambda p: p.drawPoint(pos))

    def _draw_line(self, p1, p2):
        self._paint_on_active_layer(lambda p: p.drawLine(p1, p2))

    def _paint_circle(self, center, radius):
        if center is None or radius <= 0:
            return
        ann_id = self._current_annotation_id
        if ann_id is None:
            return
        layer = self._get_or_create_layer(ann_id)
        painter = QPainter(layer)
        color = QColor(self._pen_color)
        color.setAlpha(220)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(center, radius, radius)
        painter.end()
        self._composite()
        self.signals.updated.emit()

    def _draw_circle_preview(self, radius):
        """Preview on composite copy (no layer modified)."""
        if self._circle_center is None \
                or self._circle_base_mask is None:
            return
        preview = self._circle_base_mask.copy()
        painter = QPainter(preview)
        color = QColor(self._pen_color)
        color.setAlpha(220)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(self._circle_center, radius, radius)
        painter.end()
        self.setPixmap(QPixmap.fromImage(preview))

    def _fill_connect_polygon(self):
        if len(self._connect_points) < 3:
            return
        ann_id = self._current_annotation_id
        if ann_id is None:
            return

        layer = self._get_or_create_layer(ann_id)

        # Fill polygon on top of existing content (merge, not replace)
        painter = QPainter(layer)
        color = QColor(self._pen_color)
        color.setAlpha(220)
        pen = QPen(color, self._pen_width,
                   Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QBrush(color))
        painter.drawPolygon(QPolygonF(self._connect_points))
        painter.end()
        self._composite()

    # ------------------------------------------------------------------
    # Eraser: circular brush + cursor preview
    # ------------------------------------------------------------------
    def _update_eraser_cursor(self, pos):
        """Draw eraser circle on current display (no composite)."""
        if not self._is_loaded or self._mask is None:
            return
        pixmap = QPixmap.fromImage(self._mask)
        painter = QPainter(pixmap)
        r = float(self._pen_width)
        # Black outline
        painter.setPen(QPen(QColor(0, 0, 0, 160), 2.5, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(pos), r, r)
        # White circle
        painter.setPen(QPen(QColor(255, 255, 255, 200), 1.5, Qt.SolidLine))
        painter.drawEllipse(QPointF(pos), r, r)
        # Center dot
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 150))
        painter.drawEllipse(QPointF(pos), 1.5, 1.5)
        painter.end()
        self.setPixmap(pixmap)

    def _clear_eraser_cursor(self):
        """Restore display to clean composite (no cursor circle)."""
        if self._mask is not None:
            self.setPixmap(QPixmap.fromImage(self._mask))

    def _erase_at(self, pos):
        """Erase a circular area at pos on the active layer."""
        ann_id = self._current_annotation_id
        if ann_id is None:
            return
        layer = self._get_or_create_layer(ann_id)
        painter = QPainter(layer)
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.white)
        r = float(self._pen_width)
        painter.drawEllipse(QPointF(pos), r, r)
        painter.end()

    def _erase_line(self, p1, p2):
        """Erase along line p1→p2 with circular stamps on layer."""
        ann_id = self._current_annotation_id
        if ann_id is None:
            return
        layer = self._get_or_create_layer(ann_id)
        painter = QPainter(layer)
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.white)
        r = float(self._pen_width)

        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        dist = math.hypot(dx, dy)
        if dist < 0.5:
            painter.drawEllipse(QPointF(p2), r, r)
        else:
            step = max(r * 0.4, 1.0)
            steps = max(1, int(dist / step))
            for i in range(steps + 1):
                t = i / steps
                painter.drawEllipse(
                    QPointF(p1.x() + dx * t, p1.y() + dy * t), r, r)

        painter.end()

    # ------------------------------------------------------------------
    # Magic Wand tool
    # ------------------------------------------------------------------

    def set_wand_tolerance(self, tolerance):
        self._wand_tolerance = int(tolerance)

    def set_rgb_preview(self, qimage):
        """Store RGB preview as numpy for wand flood fill."""
        if qimage is None:
            self._rgb_array = None
            return
        rgba = qimage.convertToFormat(QImage.Format_RGBA8888)
        w, h = rgba.width(), rgba.height()
        ptr = rgba.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
        self._rgb_array = arr[:, :, :3].copy()

    def _wand_select(self, pos):
        """Flood fill on RGB preview at pos with tolerance.
        Returns list of flat polygon lists or empty list."""
        if self._rgb_array is None:
            log.warning("[wand] No RGB preview available")
            return []
        if self._current_annotation_id is None:
            log.warning("[wand] No active annotation")
            return []

        h, w = self._rgb_array.shape[:2]
        x, y = int(pos.x()), int(pos.y())
        if not (0 <= x < w and 0 <= y < h):
            return []

        try:
            import cv2
            return self._wand_cv2(x, y, w, h)
        except ImportError:
            return self._wand_numpy(x, y, w, h)

    def _wand_cv2(self, x, y, w, h):
        """Wand using cv2.floodFill (fast)."""
        import cv2

        tolerance = self._wand_tolerance
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        rgb_copy = self._rgb_array.copy()

        cv2.floodFill(
            rgb_copy, mask, (x, y), 255,
            loDiff=(tolerance, tolerance, tolerance),
            upDiff=(tolerance, tolerance, tolerance),
            flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8),
        )
        selection = mask[1:-1, 1:-1] > 0

        if not np.any(selection):
            return []

        # Contour detect
        contours, _ = cv2.findContours(
            selection.astype(np.uint8),
            cv2.RETR_TREE,
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

    def _wand_numpy(self, x, y, w, h):
        """Wand fallback using numpy BFS (slower)."""
        tolerance = self._wand_tolerance
        target = self._rgb_array[y, x].astype(np.float32)
        rgb = self._rgb_array.astype(np.float32)

        # Color distance
        dist = np.sqrt(np.sum((rgb - target) ** 2, axis=2))
        candidate = dist <= tolerance * np.sqrt(3)

        # BFS from click point
        visited = np.zeros((h, w), dtype=bool)
        stack = [(y, x)]
        visited[y, x] = True
        pixels = []

        while stack:
            cy, cx = stack.pop()
            if not candidate[cy, cx]:
                continue
            pixels.append((cx, cy))
            for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if (0 <= ny < h and 0 <= nx < w
                        and not visited[ny, nx]
                        and candidate[ny, nx]):
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if len(pixels) < 3:
            return []

        # Bbox polygon fallback
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return [[
            float(x0), float(y0), float(x1), float(y0),
            float(x1), float(y1), float(x0), float(y1),
        ]]

    def _erase_composite(self):
        """Fast composite: merge only active layer change into current mask.
        Avoids full _composite() during drag for performance."""
        ann_id = self._current_annotation_id
        if ann_id is None:
            return
        layer = self._layers.get(ann_id)
        if layer is None:
            return

        # Read layer pixels
        w, h = self._layer_w, self._layer_h
        rgba = layer.convertToFormat(QImage.Format_RGBA8888)
        ptr = rgba.bits()
        ptr.setsize(h * w * 4)
        layer_arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))

        # Read current mask pixels
        mask_rgba = self._mask.convertToFormat(QImage.Format_RGBA8888)
        ptr_m = mask_rgba.bits()
        ptr_m.setsize(h * w * 4)
        mask_arr = np.frombuffer(ptr_m, np.uint8).reshape(
            (h, w, 4)).copy()

        # Update mask: where layer has content, use layer; otherwise keep mask
        has_content = layer_arr[:, :, 3] > 0
        mask_arr[has_content] = layer_arr[has_content]

        self._mask = QImage(
            mask_arr.data, w, h, w * 4, QImage.Format_RGBA8888
        ).copy().convertToFormat(QImage.Format_ARGB32)
    def _connect_click(self, pos):
        if self._connect_last is None and self._connect_points:
            self._connect_points = []
            self._connect_start = None
        point = QPointF(pos)
        if self._connect_last is None:
            self._connect_start = QPointF(point)
            self._connect_last = QPointF(point)
            self._connect_points = [QPointF(point)]
            self._drawing = True
            self._draw_dot(point)
            self.signals.updated.emit()
            return
        if len(self._connect_points) >= 3:
            dx = point.x() - self._connect_start.x()
            dy = point.y() - self._connect_start.y()
            if math.hypot(dx, dy) <= max(5.0, self._pen_width):
                self._draw_line(self._connect_last,
                                self._connect_start)
                self._connect_points.append(
                    QPointF(self._connect_start))
                self._fill_connect_polygon()
                self._emit_connect_shape_completed()
                self._connect_start = None
                self._connect_last = None
                self._connect_points = []
                self._drawing = False
                self.signals.updated.emit()
                self.signals.shape_closed.emit()
                log.debug("Connect path auto-closed")
                return
        self._draw_line(self._connect_last, point)
        self._connect_last = QPointF(point)
        self._connect_points.append(QPointF(point))
        self.signals.updated.emit()

    def _close_connect_path(self):
        if self._connect_last is None \
                or self._connect_start is None:
            self._connect_start = None
            self._connect_last = None
            self._connect_points = []
            self._drawing = False
            return
        if self._connect_last != self._connect_start:
            self._draw_line(self._connect_last,
                            self._connect_start)
        n_pts = len(self._connect_points)
        if n_pts >= 3:
            self._fill_connect_polygon()
            self._emit_connect_shape_completed()
        self._connect_start = None
        self._connect_last = None
        self._connect_points = []
        self._drawing = False
        self.signals.updated.emit()
        self.signals.shape_closed.emit()
        log.debug("Connect path closed (%d pts)", n_pts)

    def _emit_connect_shape_completed(self):
        pts = list(self._connect_points)
        if len(pts) < 3:
            return
        bbox = self._compute_polygon_bbox(pts)
        seg = self._flatten_points(pts)
        area = self._polygon_pixel_area(pts)
        self.signals.shape_completed.emit(
            self._current_annotation_id, bbox, seg, area)

    # ------------------------------------------------------------------
    # Flood fill
    # ------------------------------------------------------------------

    def flood_fill(self, pos):
        x0, y0 = int(pos.x()), int(pos.y())
        ann_id = self._current_annotation_id
        if ann_id is None:
            return
        layer = self._get_or_create_layer(ann_id)
        w, h = self._layer_w, self._layer_h

        rgba = layer.convertToFormat(QImage.Format_RGBA8888)
        ptr = rgba.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape(
            (h, w, 4)).copy()

        if not (0 <= x0 < w and 0 <= y0 < h):
            return
        target_color = tuple(arr[y0, x0])
        color = self._pen_color
        fill_color = np.array(
            [color.red(), color.green(), color.blue(), 220],
            dtype=np.uint8)
        if target_color == tuple(fill_color):
            return

        filled_pixels = []
        queue = deque([(y0, x0)])
        while queue:
            y, x = queue.popleft()
            if tuple(arr[y, x]) != target_color:
                continue
            arr[y, x] = fill_color
            filled_pixels.append((x, y))
            if x > 0:       queue.append((y, x - 1))
            if x < w - 1:   queue.append((y, x + 1))
            if y > 0:       queue.append((y - 1, x))
            if y < h - 1:   queue.append((y + 1, x))

        new_layer = QImage(
            arr.data, w, h, w * 4, QImage.Format_RGBA8888
        ).copy().convertToFormat(QImage.Format_ARGB32)
        self._layers[ann_id] = new_layer
        self._composite()
        self.signals.updated.emit()

        if filled_pixels:
            xs = [p[0] for p in filled_pixels]
            ys = [p[1] for p in filled_pixels]
            xm, xM = int(min(xs)), int(max(xs))
            ym, yM = int(min(ys)), int(max(ys))
            bbox = [xm, ym, xM - xm + 1, yM - ym + 1]
            seg = [float(xm), float(ym), float(xM), float(ym),
                   float(xM), float(yM), float(xm), float(yM)]
            self.signals.shape_completed.emit(
                self._current_annotation_id, bbox, seg,
                len(filled_pixels))
        self.signals.shape_closed.emit()

    # ------------------------------------------------------------------
    # Layer-level operations (called by window.py)
    # ------------------------------------------------------------------

    def paint_annotation_layer(self, ann_id, polygons, color_rgb):
        """Paint polygons onto an annotation's layer.
        Uses OddEvenFill rule so inner contours (holes) are preserved.
        """
        layer = self._get_or_create_layer(ann_id)
        painter = QPainter(layer)
        color = QColor(*color_rgb, 220)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))

        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        for poly in polygons:
            pts = self._segmentation_to_points(poly)
            if len(pts) >= 3:
                path.addPolygon(QPolygonF(pts))
                path.closeSubpath()

        if not path.isEmpty():
            path.setFillRule(Qt.OddEvenFill)
            painter.drawPath(path)

        painter.end()
        self._composite()

    def clear_annotation_layer(self, ann_id):
        layer = self._layers.get(ann_id)
        if layer is not None:
            layer.fill(0)
        self._composite()

    def recolor_annotation_layer(self, ann_id, old_color, new_color):
        layer = self._layers.get(ann_id)
        if layer is None:
            return
        w, h = self._layer_w, self._layer_h
        rgba = layer.convertToFormat(QImage.Format_RGBA8888)
        ptr = rgba.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape(
            (h, w, 4)).copy()
        match = _match_color(arr, QColor(*old_color, alpha=255))
        if not np.any(match):
            return
        arr[match, 0] = new_color[0]
        arr[match, 1] = new_color[1]
        arr[match, 2] = new_color[2]
        self._layers[ann_id] = QImage(
            arr.data, w, h, w * 4, QImage.Format_RGBA8888
        ).copy().convertToFormat(QImage.Format_ARGB32)
        self._composite()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _segmentation_to_points(segmentation):
        points = []
        it = iter(segmentation)
        for x, y in zip(it, it):
            points.append(QPointF(float(x), float(y)))
        return points

    @staticmethod
    def _compute_polygon_bbox(points):
        xs = [p.x() for p in points]
        ys = [p.y() for p in points]
        x0, y0 = min(xs), min(ys)
        return [x0, y0, max(xs) - x0, max(ys) - y0]

    @staticmethod
    def _flatten_points(points):
        flat = []
        for p in points:
            flat.extend([float(p.x()), float(p.y())])
        return flat

    @staticmethod
    def _polygon_pixel_area(points):
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i].x() * points[j].y()
            area -= points[j].x() * points[i].y()
        return int(abs(area) / 2.0)

    @staticmethod
    def _circle_to_polygon(center, radius, n_points=36):
        cx, cy = center.x(), center.y()
        pts = []
        for i in range(n_points):
            angle = 2.0 * math.pi * i / n_points
            pts.append(QPointF(cx + radius * math.cos(angle),
                               cy + radius * math.sin(angle)))
        return pts

    @staticmethod
    def _normalize_segmentation(segmentation):
        if not segmentation:
            return []
        if isinstance(segmentation[0], (int, float)):
            return [segmentation]
        return [p for p in segmentation if p]
    def set_layers_visible_batch(self, visibility_map):
        """Batch update visibility for multiple annotations.
        visibility_map = {ann_id: bool, ...}
        Composites only ONCE at the end.
        """
        for ann_id, visible in visibility_map.items():
            self._layer_visible[int(ann_id)] = bool(visible)
        self._composite()