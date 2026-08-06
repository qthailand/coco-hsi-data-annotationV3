# hsi_annotation/registry.py
"""
registry.py
-----------
CategoryRegistry  — dict store  {category_id: {"name": str, "color": (R,G,B), "visible": bool}}
AnnotationRegistry — list store [{"id": int, "category_id": int, "bbox": [...],
                                   "segmentation": [...], "area": int, "visible": bool}]

Both emit Qt signals on every mutation so the UI stays in sync.
"""

import logging

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor

log = logging.getLogger(__name__)

_DEFAULT_COLORS = [
    (231,  76,  60),
    ( 46, 204, 113),
    ( 52, 152, 219),
    (241, 196,  15),
    (155,  89, 182),
    ( 26, 188, 156),
    (230, 126,  34),
    ( 52,  73,  94),
    (236, 112, 176),
    (149, 165, 166),
]


# ======================================================================
# CategoryRegistry
# ======================================================================

class CategoryRegistry(QObject):
    """
    Signals: category_added(id), category_about_to_be_removed(id, color_tuple),
             category_removed(id), category_changed(id),
             category_color_changed(id, old_color, new_color), reset()
    """
    category_added               = pyqtSignal(int)
    category_about_to_be_removed = pyqtSignal(int, object)
    category_removed             = pyqtSignal(int)
    category_changed             = pyqtSignal(int)
    category_color_changed       = pyqtSignal(int, object, object)
    reset                        = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._categories = {}
        self._next_id = 1

    # --- read ---

    def ids(self):
        return sorted(self._categories.keys())

    def get(self, category_id):
        return self._categories.get(category_id)

    def name(self, category_id):
        e = self._categories.get(category_id)
        return e["name"] if e else ""

    def color(self, category_id):
        e = self._categories.get(category_id)
        return e["color"] if e else (128, 128, 128)

    def qcolor(self, category_id, alpha=220):
        r, g, b = self.color(category_id)
        return QColor(r, g, b, alpha)

    def is_visible(self, category_id):
        e = self._categories.get(category_id)
        return e["visible"] if e else True

    def __len__(self):
        return len(self._categories)

    def __contains__(self, category_id):
        return category_id in self._categories

    def as_list(self):
        """[(id, name, QColor), ...] sorted by id."""
        return [(cid, self._categories[cid]["name"], self.qcolor(cid))
                for cid in self.ids()]

    # --- mutations ---

    def add_category(self, name=None, color=None, category_id=None):
        if category_id is not None:
            if category_id in self._categories:
                raise ValueError("category_id {} already exists".format(category_id))
            cid = category_id
            self._next_id = max(self._next_id, cid + 1)
        else:
            cid = self._next_id
            self._next_id += 1

        if name is None:
            name = "Category {}".format(cid)
        if color is None:
            color = _DEFAULT_COLORS[(cid - 1) % len(_DEFAULT_COLORS)]

        self._categories[cid] = {"name": name, "color": tuple(color), "visible": True}
        self.category_added.emit(cid)
        return cid

    def remove_category(self, category_id):
        if category_id not in self._categories:
            return False
        old_color = tuple(self._categories[category_id]["color"])
        old_name = self._categories[category_id]["name"]
        log.info("Removing category id=%d  name='%s'  color=%s",
                 category_id, old_name, old_color)
        self.category_about_to_be_removed.emit(category_id, old_color)
        del self._categories[category_id]
        self.category_removed.emit(category_id)
        return True

    def set_name(self, category_id, name):
        if category_id not in self._categories:
            return False
        self._categories[category_id]["name"] = str(name)
        self.category_changed.emit(category_id)
        return True

    def set_color(self, category_id, color):
        if category_id not in self._categories:
            return False
        if isinstance(color, QColor):
            color = (color.red(), color.green(), color.blue())
        old_color = tuple(self._categories[category_id]["color"])
        new_color = tuple(int(v) for v in color)
        self._categories[category_id]["color"] = new_color
        self.category_changed.emit(category_id)
        if old_color != new_color:
            self.category_color_changed.emit(category_id, old_color, new_color)
        return True

    def set_visible(self, category_id, visible):
        if category_id not in self._categories:
            return False
        self._categories[category_id]["visible"] = bool(visible)
        self.category_changed.emit(category_id)
        return True

    def clear(self):
        self._categories.clear()
        self._next_id = 1
        self.reset.emit()

    def load(self, data):
        self._categories.clear()
        self._next_id = 1
        for cid, entry in data.items():
            cid = int(cid)
            self._categories[cid] = {
                "name":    str(entry.get("name", "Category {}".format(cid))),
                "color":   tuple(int(v) for v in entry.get("color", (128, 128, 128))),
                "visible": bool(entry.get("visible", True)),
            }
            self._next_id = max(self._next_id, cid + 1)
        self.reset.emit()

    def to_dict(self):
        return {cid: dict(e) for cid, e in self._categories.items()}

    # backward-compat aliases (so old imports keep working during transition)
    add_label    = lambda self, *a, **kw: self.add_category(*a, **kw)
    remove_label = lambda self, *a, **kw: self.remove_category(*a, **kw)


# ======================================================================
# AnnotationRegistry
# ======================================================================

class AnnotationRegistry(QObject):
    """
    Signals: annotation_added(id), annotation_removed(id),
             annotation_changed(id), annotation_visibility_changed(id), reset()
    """
    annotation_added              = pyqtSignal(int)
    annotation_removed            = pyqtSignal(int)
    annotation_changed            = pyqtSignal(int)
    annotation_visibility_changed = pyqtSignal(int)
    reset                         = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._annotations = []
        self._next_id = 1

    def __len__(self):
        return len(self._annotations)

    def __iter__(self):
        return iter(self._annotations)

    def get(self, annotation_id):
        for a in self._annotations:
            if a["id"] == annotation_id:
                return a
        return None

    def by_category(self, category_id):
        return [a for a in self._annotations
                if a["category_id"] == category_id]

    def all(self):
        return list(self._annotations)

    def add(self, category_id, bbox=None, segmentation=None, area=0):
        ann_id = self._next_id
        self._next_id += 1
        self._annotations.append({
            "id":            ann_id,
            "category_id":   category_id,
            "bbox":          list(bbox or []),
            "segmentation":  list(segmentation or []),
            "area":          area,
            "visible":       True,
        })
        self.annotation_added.emit(ann_id)
        return ann_id

    def remove(self, annotation_id):
        for i, a in enumerate(self._annotations):
            if a["id"] == annotation_id:
                self._annotations.pop(i)
                self.annotation_removed.emit(annotation_id)
                return True
        return False

    def remove_by_category(self, category_id):
        ids = [a["id"] for a in self._annotations
               if a["category_id"] == category_id]
        for ann_id in ids:
            self.remove(ann_id)

    def update(self, annotation_id, **kwargs):
        a = self.get(annotation_id)
        if a is None:
            return False
        for k, v in kwargs.items():
            if k != "id":
                a[k] = v
        self.annotation_changed.emit(annotation_id)
        return True

    def set_visible(self, annotation_id, visible):
        a = self.get(annotation_id)
        if a is None:
            return False
        a["visible"] = bool(visible)
        self.annotation_visibility_changed.emit(annotation_id)
        return True

    def is_visible(self, annotation_id):
        a = self.get(annotation_id)
        return a["visible"] if a else True

    def clear(self):
        self._annotations.clear()
        self._next_id = 1
        self.reset.emit()

    def load(self, data):
        self._annotations.clear()
        self._next_id = 1
        for entry in data:
            ann = {
                "id":            int(entry.get("id", self._next_id)),
                "category_id":   int(entry.get("category_id",
                                               entry.get("label_id", 0))),
                "bbox":          list(entry.get("bbox", [])),
                "segmentation":  list(entry.get("segmentation", [])),
                "area":          int(entry.get("area", 0)),
                "visible":       bool(entry.get("visible", True)),
            }
            self._annotations.append(ann)
            self._next_id = max(self._next_id, ann["id"] + 1)
        self.reset.emit()

    def to_list(self):
        return [dict(a) for a in self._annotations]
    def clear_for_switch(self):
        """Clear annotations without resetting _next_id.
        Used when switching cubes — ID counter must persist."""
        self._annotations.clear()
        self.reset.emit()

    def to_state(self):
        """Snapshot current state for cache."""
        return {
            "annotations": [dict(a) for a in self._annotations],
            "next_id": self._next_id,
        }

    def from_state(self, state):
        """Restore from a cached snapshot."""
        self._annotations = [dict(a) for a in state["annotations"]]
        self._next_id = state["next_id"]
        self.reset.emit()