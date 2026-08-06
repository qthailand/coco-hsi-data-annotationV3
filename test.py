# test_draw_annotations.py
"""
Test script: Load a single-cube COCO JSON + RGB preview,
draw annotations, show preview, save *_annotated.png.

Usage:
    python test_draw_annotations.py <path_to_coco_json> [--alpha 0.35] [--no-show]
"""

import sys
import os
import json
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV required.  pip install opencv-python")
    sys.exit(1)


DEFAULT_COLORS = [
    (231,  76,  60),
    (46, 204, 113),
    (52, 152, 219),
    (241, 196,  15),
    (155,  89, 182),
    (26, 188, 156),
    (230, 126,  34),
    (52,  73,  94),
    (236, 112, 176),
    (149, 165, 166),
]


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_rgb(json_path):
    """Find *_rgb.png next to the JSON file."""
    json_dir = os.path.dirname(os.path.abspath(json_path))
    stem = os.path.splitext(os.path.basename(json_path))[0]

    candidates = [
        os.path.join(json_dir, f"{stem}_rgb.png"),
        os.path.join(json_dir, f"{stem}_rgb.jpg"),
    ]

    # Also strip _gt / _coco suffix if present
    for suffix in ["_gt", "_coco", "_coco_gt"]:
        if stem.endswith(suffix):
            base = stem[:len(stem) - len(suffix)]
            candidates.append(os.path.join(json_dir, f"{base}_rgb.png"))
            candidates.append(os.path.join(json_dir, f"{base}_rgb.jpg"))

    for c in candidates:
        if os.path.isfile(c):
            return c

    # Search directory
    for name in os.listdir(json_dir):
        if name.endswith("_rgb.png") or name.endswith("_rgb.jpg"):
            return os.path.join(json_dir, name)

    return None


def build_cat_color_map(categories):
    color_map = {}
    for idx, cat in enumerate(categories):
        cid = int(cat["id"])
        r, g, b = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        color_map[cid] = (b, g, r)
    return color_map


def draw_annotations(img, annotations, cat_name, color_map, alpha=0.35):
    overlay = img.copy()

    for ann in annotations:
        ann_id = ann.get("id", 0)
        cat_id = int(ann.get("category_id", 0))
        segs = ann.get("segmentation", [])
        bbox = ann.get("bbox", [])
        name = cat_name.get(cat_id, f"?{cat_id}")
        color = color_map.get(cat_id, (200, 200, 200))

        # Fill polygon
        if segs:
            polygons = segs if isinstance(segs[0], list) else [segs]
            for poly in polygons:
                pts = []
                it = iter(poly)
                for x, y in zip(it, it):
                    pts.append((int(round(float(x))),
                                int(round(float(y)))))
                if len(pts) >= 3:
                    cv2.fillPoly(overlay,
                                 [np.array(pts, dtype=np.int32)], color)

        # Polygon outline
        if segs:
            polygons = segs if isinstance(segs[0], list) else [segs]
            for poly in polygons:
                pts = []
                it = iter(poly)
                for x, y in zip(it, it):
                    pts.append((int(round(float(x))),
                                int(round(float(y)))))
                if len(pts) >= 3:
                    cv2.polylines(img,
                                  [np.array(pts, dtype=np.int32)],
                                  True, color, 2)

        # Bbox
        if len(bbox) >= 4:
            x0 = int(round(float(bbox[0])))
            y0 = int(round(float(bbox[1])))
            bw = int(round(float(bbox[2])))
            bh = int(round(float(bbox[3])))
            cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh),
                          color, 2)

        # Label
        if len(bbox) >= 4:
            x0 = int(round(float(bbox[0])))
            y0 = int(round(float(bbox[1])))
            label = f"#{ann_id} ({name})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), _ = cv2.getTextSize(
                label, font, font_scale, thickness)
            pad = 3
            py1 = max(0, y0 - th - pad * 2)
            cv2.rectangle(img, (x0, py1),
                          (x0 + tw + pad * 2, y0),
                          color, cv2.FILLED)
            cv2.putText(img, label, (x0 + pad, y0 - pad),
                        font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def fit_to_screen(img, max_w=1600, max_h=900):
    h, w = img.shape[:2]
    if w <= max_w and h <= max_h:
        return img, 1.0
    scale = min(max_w / w, max_h / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA), scale


def main():
    parser = argparse.ArgumentParser(
        description="Draw & preview COCO annotations")
    parser.add_argument("json_path", nargs="?",
                        default="./c02_calibrated.bsq.json",
                        help="Path to COCO JSON")
    parser.add_argument("--alpha", type=float, default=0.35,
                        help="Polygon fill transparency")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip preview window")
    args = parser.parse_args()

    json_path = args.json_path
    alpha = args.alpha

    if not os.path.isfile(json_path):
        print(f"ERROR: File not found: {json_path}")
        sys.exit(1)

    coco = load_coco(json_path)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    cat_name_map = {int(c["id"]): c.get("name", "?")
                    for c in categories}
    color_map = build_cat_color_map(categories)

    # Find RGB
    rgb_path = find_rgb(json_path)
    if rgb_path is None:
        print(f"ERROR: No *_rgb.png found next to {json_path}")
        sys.exit(1)

    print(f"JSON       : {json_path}")
    print(f"RGB        : {rgb_path}")
    print(f"Images     : {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Categories : {len(categories)}")
    print()

    img = cv2.imread(rgb_path)
    if img is None:
        print(f"ERROR: Cannot read image: {rgb_path}")
        sys.exit(1)

    print(f"Image size : {img.shape[1]}x{img.shape[0]}")

    # Draw
    result = draw_annotations(img, annotations,
                              cat_name_map, color_map, alpha)

    # Save
    out_path = os.path.splitext(json_path)[0] + "_annotated.png"
    cv2.imwrite(out_path, result)
    print(f"Saved      : {out_path}")

    # Summary
    counts = {}
    areas = {}
    for ann in annotations:
        cid = int(ann.get("category_id", 0))
        cname = cat_name_map.get(cid, f"?{cid}")
        counts[cname] = counts.get(cname, 0) + 1
        areas[cname] = areas.get(cname, 0) + int(
            ann.get("area", 0))

    if counts:
        print(f"\n{'Category':<15} {'Count':>6} {'Area':>10}")
        print(f"{'-'*15} {'-'*6} {'-'*10}")
        for cname in sorted(counts.keys()):
            print(f"{cname:<15} {counts[cname]:>6} "
                  f"{areas[cname]:>10}")

    # Preview
    if not args.no_show:
        display, _ = fit_to_screen(result)

        # Info bar
        bar_h = 40
        bar = np.zeros((bar_h, display.shape[1], 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)
        info = (f"{os.path.basename(json_path)}  "
                f"| {img.shape[1]}x{img.shape[0]}  "
                f"| {len(annotations)} anns  "
                f"| {len(categories)} cat(s)  "
                f"| Q/ESC=quit")
        cv2.putText(bar, info, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)
        display_with_bar = np.vstack([bar, display])

        window_name = "Annotation Preview"
        cv2.imshow(window_name, display_with_bar)
        print("\nPress Q or ESC to close preview window.")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) != 1:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
