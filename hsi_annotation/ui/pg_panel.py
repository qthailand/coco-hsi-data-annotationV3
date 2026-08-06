#hsi_annotation/ui/pg_panel.py
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)


class PgPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        left_axis = pg.AxisItem(orientation='left')
        left_axis.enableAutoSIPrefix(False)
        bottom_axis = pg.AxisItem(orientation='bottom')
        bottom_axis.enableAutoSIPrefix(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._spec_plot = pg.PlotWidget(
            axisItems={'left': left_axis, 'bottom': bottom_axis})
        self._spec_plot.setBackground("#1e1e1e")
        self._spec_plot.setLabel("left", "Reflectance")
        self._spec_plot.setLabel("bottom", "Band index")
        self._spec_plot.showGrid(x=True, y=True, alpha=0.25)
        self._spec_plot.addLegend(offset=(10, 10))
        self._spec_curve = self._spec_plot.plot(
            [], [], pen=pg.mkPen((255, 255, 255), width=1.5),
            name="At cursor")
        self._class_curves = {}
        layout.addWidget(self._spec_plot)

        self._spec_lbl = QLabel("Click on canvas to probe spectrum")
        self._spec_lbl.setAlignment(Qt.AlignCenter)
        self._spec_lbl.setStyleSheet(
            "font-size:10px; color:#888; padding:2px;")
        layout.addWidget(self._spec_lbl)

        self._progress_lbl = QLabel("")
        self._progress_lbl.setAlignment(Qt.AlignCenter)
        self._progress_lbl.setStyleSheet(
            "font-size:10px; color:#a0a0a0; padding:2px;")
        layout.addWidget(self._progress_lbl)

    def update_spectrum(self, x, y, arr):
        self._spec_curve.setData(
            np.arange(len(arr), dtype=float), arr.astype(float))
        self._spec_lbl.setText(
            "cursor: col={}  row={}  bands={}  mean={:.4f}".format(
                x, y, len(arr), arr.mean()))

    def clear_cursor_spectrum(self):
        self._spec_curve.setData([], [])
        self._spec_lbl.setText("Click on canvas to probe spectrum")

    def set_spectrum_status(self, text):
        self._progress_lbl.setText(text)

    def reset_spectrum_status(self):
        self._progress_lbl.setText("")

    def update_class_spectra(self, class_data):
        self.reset_spectrum_status()
        for curve in self._class_curves.values():
            self._spec_plot.removeItem(curve)
        self._class_curves.clear()

        for name, color, avg in class_data:
            if avg is None:
                continue
            curve = self._spec_plot.plot(
                np.arange(len(avg), dtype=float),
                avg.astype(float),
                pen=pg.mkPen(
                    (color.red(), color.green(), color.blue()), width=2.2),
                name=name,
            )
            self._class_curves[name] = curve

        parts = [
            "<span style='color:#{0:02x}{1:02x}{2:02x}'>{3}</span>".format(
                color.red(), color.green(), color.blue(), name)
            for name, color, avg in class_data
            if avg is not None
        ]
        self._spec_plot.setTitle("  ".join(parts) if parts else "")