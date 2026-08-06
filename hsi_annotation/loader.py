#hsi_annotation/loader.py
import json
import os

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

# list HSI files with ground truth pairs for checking available GT
# returns a tuple (hdr_path, gt_exists, gt_files)
def dataset_path(path):
    root = os.path.abspath(path)
    gt_path = os.path.join(root, "GT")
    dt_path = os.path.join(root, "HSI")

    if not os.path.isdir(dt_path):
        raise FileNotFoundError(f"HSI directory not found: {dt_path}")
    if not os.path.isdir(gt_path):
        raise FileNotFoundError(f"GT directory not found: {gt_path}")

    gt_files = [entry.name for entry in os.scandir(gt_path) if entry.is_file()]
    gt_basenames = set()
    for name in gt_files:
        base, _ = os.path.splitext(name)
        gt_basenames.add(base)
        if base.endswith("_mask"):
            gt_basenames.add(base[: -len("_mask")])

    result = []
    for dt in os.scandir(dt_path):
        if dt.is_file() and dt.name.endswith('.hdr'):
            hdr_base = os.path.splitext(dt.name)[0]
            matches = [name for name in gt_files if name.startswith(hdr_base)]
            result.append((dt.name, bool(matches), matches))

    return result


def missing_ground_truth(path):
    return [hdr for hdr, exists, matches in dataset_path(path) if not exists]


def load_coco_json(path):
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    assert "images" in coco and len(coco["images"]) > 0, "Missing images in COCO file"
    assert "annotations" in coco, "Missing annotations in COCO file"
    return coco


def find_coco_json_files(path, recursive=False):
    path = os.path.abspath(path)
    if os.path.isfile(path) and path.lower().endswith('.json'):
        return [path]

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path not found: {path}")

    matches = []
    if recursive:
        for root, _, files in os.walk(path):
            for name in files:
                if name.lower().endswith('.json'):
                    matches.append(os.path.join(root, name))
    else:
        for name in os.listdir(path):
            if name.lower().endswith('.json'):
                matches.append(os.path.join(path, name))

    return sorted(matches)


def resolve_datacube_path(coco_json_path, file_name):
    if not file_name:
        return None

    if os.path.isabs(file_name) and os.path.exists(file_name):
        return os.path.abspath(file_name)

    base_dir = os.path.dirname(os.path.abspath(coco_json_path))
    candidate = os.path.join(base_dir, file_name)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    root, ext = os.path.splitext(file_name)
    if ext == "":
        for alt_ext in [".hdr", ".bsq", ".tif", ".tiff", ".img", ".dat"]:
            candidate = os.path.join(base_dir, root + alt_ext)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

    return None


def load_coco_json_dataset(path, recursive=False):
    files = find_coco_json_files(path, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No COCO JSON files found in: {path}")

    dataset = []
    for coco_json_path in files:
        coco = load_coco_json(coco_json_path)
        dataset.append({
            "json_path": coco_json_path,
            "coco": coco,
        })
    return dataset


def group_coco_json_dataset_by_datacube(dataset):
    groups = {}
    for entry in dataset:
        coco = entry["coco"]
        image = coco["images"][0]
        file_name = image.get("file_name")
        datacube_path = resolve_datacube_path(entry["json_path"], file_name)

        if datacube_path is None:
            group_key = f"no_cube:{os.path.abspath(entry['json_path'])}"
        else:
            group_key = datacube_path

        groups.setdefault(group_key, []).append({
            **entry,
            "datacube_path": datacube_path,
            "image_file_name": file_name,
        })
    return groups


def cube_gt_table(path, recursive=False):
    if pd is None:
        raise ImportError("pandas is required to build the cube-GT table")

    rows = []
    labels_per_cube = {}
    try:
        dataset = load_coco_json_dataset(path, recursive=recursive)
    except FileNotFoundError:
        dataset = None

    if dataset:
        groups = group_coco_json_dataset_by_datacube(dataset)
        for group_key, entries in groups.items():
            if group_key.startswith("no_cube:"):
                cube_name = os.path.basename(entries[0]["json_path"])
                datacube_path = None
            else:
                cube_name = os.path.basename(group_key)
                datacube_path = group_key

            labels = set()
            for entry in entries:
                coco = entry["coco"]
                cat_map = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
                for ann in coco.get("annotations", []):
                    cat_id = ann.get("category_id")
                    if cat_id in cat_map:
                        labels.add(cat_map[cat_id])
            labels_per_cube[cube_name] = sorted(labels)

            rows.append({
                "datacube": cube_name,
                "ground truth": bool(datacube_path),
                "annotations": sum(len(entry["coco"].get("annotations", [])) for entry in entries),
            })
    else:
        for hdr_name, exists, matches in dataset_path(path):
            rows.append({
                "datacube": hdr_name,
                "ground truth": exists,
                "annotations": len(matches),
            })

        gt_dir = os.path.join(os.path.abspath(path), "GT")
        try:
            gt_dataset = load_coco_json_dataset(gt_dir)
        except FileNotFoundError:
            gt_dataset = []
        for gt_entry in gt_dataset:
            coco = gt_entry["coco"]
            cube_name = coco["images"][0].get("file_name", os.path.basename(gt_entry["json_path"]))
            cat_map = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
            labels = {cat_map[ann["category_id"]] for ann in coco.get("annotations", []) if ann.get("category_id") in cat_map}
            labels_per_cube[cube_name] = sorted(labels)

    return pd.DataFrame(rows), labels_per_cube


def print_cube_gt_table(path, recursive=False):
    df, _ = cube_gt_table(path, recursive=recursive)
    print(df.to_string(index=False))


def list_labels_per_cube(path, recursive=False):
    """Return a dict mapping cube name -> sorted list of label names used in its annotations."""
    _, labels_per_cube = cube_gt_table(path, recursive=recursive)
    return labels_per_cube


def print_labels_per_cube(path, recursive=False):
    """Print annotation labels used in each datacube."""
    labels_per_cube = list_labels_per_cube(path, recursive=recursive)
    for cube_name, labels in labels_per_cube.items():
        print(f"\n{cube_name}:")
        for label in labels:
            print(f"  - {label}")

def list_hdr_files(path):
    """Return a sorted list of .hdr file paths in the given directory."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    files = []
    for name in sorted(os.listdir(path)):
        if name.lower().endswith(".hdr"):
            full = os.path.join(path, name)
            if os.path.isfile(full):
                files.append(full)
    return files