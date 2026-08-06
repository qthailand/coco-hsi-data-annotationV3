# test_draw_annotations.py
"""
Test script: Load a workspace-level COCO JSON + RGB preview images,
draw all annotations, show preview window, save *_annotated.png.

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


def find_rgb_for_image(image_entry, json_dir):
    fname = image_entry.get("file_name", "")
    # Strip .bsq.hdr / .hdr → stem
    stem = fname
    for ext in [".bsq.hdr", ".hdr", ".HDR"]:
        if stem.lower().endswith(ext.lower()):
            stem = stem[:len(stem) - len(ext)]
            break

    search_dirs = [
        json_dir,
        os.path.join(json_dir, "rgb"),
        os.path.join(json_dir, "previews"),
        os.path.join(json_dir, "GT"),
    ]

    candidates = [
        f"{stem}_rgb.png",
        f"{stem}_rgb.jpg",
        f"{stem}.png",
        f"{stem}.jpg",
    ]

    for search_dir in search_dirs:
        for candidate in candidates:
            path = os.path.join(search_dir, candidate)
            if os.path.isfile(path):
                return path

    if os.path.isabs(fname):
        abs_stem = fname
        for ext in [".bsq.hdr", ".hdr"]:
            if abs_stem.lower().endswith(ext.lower()):
                abs_stem = abs_stem[:len(abs_stem) - len(ext)]
                break
        for ext in ["_rgb.png", "_rgb.jpg", ".png", ".jpg"]:
            if os.path.isfile(abs_stem + ext):
                return abs_stem + ext

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
                    cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)],
                                 color)

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
                    cv2.polylines(img, [np.array(pts, dtype=np.int32)],
                                  True, color, 2)

        # Bbox
        if len(bbox) >= 4:
            x0, y0 = int(round(float(bbox[0]))), int(round(float(bbox[1])))
            bw, bh = int(round(float(bbox[2]))), int(round(float(bbox[3])))
            cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), color, 2)

        # Label
        if len(bbox) >= 4:
            x0, y0 = int(round(float(bbox[0]))), int(round(float(bbox[1])))
            label = f"#{ann_id} ({name})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            pad = 3
            py1 = max(0, y0 - th - pad * 2)
            cv2.rectangle(img, (x0, py1), (x0 + tw + pad * 2, y0),
                          color, cv2.FILLED)
            cv2.putText(img, label, (x0 + pad, y0 - pad),
                        font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def fit_to_screen(img, max_w=1600, max_h=900):
    """Resize image to fit screen while keeping aspect ratio."""
    h, w = img.shape[:2]
    if w <= max_w and h <= max_h:
        return img, 1.0
    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def print_summary(annotations, cat_name_map, indent="    "):
    counts = {}
    areas = {}
    for ann in annotations:
        cid = int(ann.get("category_id", 0))
        cname = cat_name_map.get(cid, f"?{cid}")
        counts[cname] = counts.get(cname, 0) + 1
        areas[cname] = areas.get(cname, 0) + int(ann.get("area", 0))
    if counts:
        print(f"{indent}{'Category':<15} {'Count':>6} {'Area':>10}")
        print(f"{indent}{'-'*15} {'-'*6} {'-'*10}")
        for cname in sorted(counts.keys()):
            print(f"{indent}{cname:<15} {counts[cname]:>6} {areas[cname]:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Draw & preview COCO annotations on RGB images")
    parser.add_argument("json_path", nargs="?",
                        default="./20260331.json",
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

    json_dir = os.path.dirname(os.path.abspath(json_path))
    coco = load_coco(json_path)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    cat_name_map = {int(c["id"]): c.get("name", "?") for c in categories}
    color_map = build_cat_color_map(categories)

    print(f"JSON       : {json_path}")
    print(f"Images     : {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Categories : {len(categories)}")
    print(f"Alpha      : {alpha}")
    print()

    anns_by_image = {}
    for ann in annotations:
        img_id = int(ann.get("image_id", 0))
        anns_by_image.setdefault(img_id, []).append(ann)

    total_annotated = 0
    total_skipped = 0

    window_name = "Annotations Preview  [Space=next  Q/ESC=quit]"
    show_preview = not args.no_show

    for img_idx, img_entry in enumerate(images):
        img_id = int(img_entry.get("id", 0))
        fname = img_entry.get("file_name", "")
        w = img_entry.get("width", 0)
        h = img_entry.get("height", 0)
        img_anns = anns_by_image.get(img_id, [])

        print(f"{'='*50}")
        print(f"[{img_idx+1}/{len(images)}] {fname}")
        print(f"  Size: {w}x{h}  Annotations: {len(img_anns)}")

        rgb_path = find_rgb_for_image(img_entry, json_dir)
        if rgb_path is None:
            print(f"  RGB preview: NOT FOUND — skipped")
            total_skipped += 1
            continue

        print(f"  RGB preview: {os.path.basename(rgb_path)}")

        img = cv2.imread(rgb_path)
        if img is None:
            print(f"  ERROR: Cannot read image — skipped")
            total_skipped += 1
            continue

        print(f"  Actual image: {img.shape[1]}x{img.shape[0]}")

        if img_anns:
            result = draw_annotations(img.copy(), img_anns,
                                      cat_name_map, color_map, alpha)
        else:
            result = img.copy()

        # Save
        out_stem = fname
        for ext in [".bsq.hdr", ".hdr"]:
            if out_stem.lower().endswith(ext.lower()):
                out_stem = out_stem[:len(out_stem) - len(ext)]
                break
        out_path = os.path.join(json_dir, f"{out_stem}_annotated.png")
        cv2.imwrite(out_path, result)
        print(f"  Saved: {os.path.basename(out_path)}")
        total_annotated += 1

        print_summary(img_anns, cat_name_map)

        # Preview
        if show_preview:
            display, scale = fit_to_screen(result)

            # Add info bar at top
            bar_h = 40
            bar = np.zeros((bar_h, display.shape[1], 3), dtype=np.uint8)
            bar[:] = (40, 40, 40)
            info = (f"[{img_idx+1}/{len(images)}] {fname}  "
                    f"| {img.shape[1]}x{img.shape[0]}  "
                    f"| {len(img_anns)} anns  "
                    f"| Space=next  Q=quit")
            cv2.putText(bar, info, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1, cv2.LINE_AA)
            display_with_bar = np.vstack([bar, display])

            cv2.imshow(window_name, display_with_bar)

            # Wait for key
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord(' ') or key == ord('\r') or key == 13:
                    break  # next image
                elif key == ord('q') or key == ord('Q') or key == 27:
                    # Quit
                    cv2.destroyAllWindows()
                    _print_final(images, annotations, cat_name_map,
                                 total_annotated, total_skipped)
                    return
                elif key == ord('s'):
                    # Save and continue
                    break

    if show_preview:
        cv2.destroyAllWindows()

    _print_final(images, annotations, cat_name_map,
                 total_annotated, total_skipped)


def _print_final(images, annotations, cat_name_map,
                 total_annotated, total_skipped):
    print()
    print("=" * 50)
    print(f"Annotated: {total_annotated} / {len(images)} images")
    if total_skipped:
        print(f"Skipped  : {total_skipped} (no RGB preview)")

    g_counts, g_areas = {}, {}
    for ann in annotations:
        cid = int(ann.get("category_id", 0))
        cname = cat_name_map.get(cid, f"?{cid}")
        g_counts[cname] = g_counts.get(cname, 0) + 1
        g_areas[cname] = g_areas.get(cname, 0) + int(ann.get("area", 0))
    if g_counts:
        print(f"\n{'Category':<15} {'Count':>6} {'Total px':>10}")
        print(f"{'-'*15} {'-'*6} {'-'*10}")
        for cname in sorted(g_counts.keys()):
            print(f"{cname:<15} {g_counts[cname]:>6} {g_areas[cname]:>10}")


if __name__ == "__main__":
    main()
