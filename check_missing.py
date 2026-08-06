# -*- coding: utf-8 -*-
"""
List hyperspectral datacubes that do not yet have a ground truth file.
"""

import os
from hsi_annotation import loader


root_path = os.path.abspath(r'C:\Datacubes\20260429\Calibrated\subregion')

df, labels = loader.cube_gt_table(root_path)
print(df.to_string(index=False))

print("\n--- Annotation labels per cube ---")
for cube_name, cube_labels in labels.items():
    print(f"\n{cube_name}:")
    for label in cube_labels:
        ok = 'True' if cube_name.startswith(label[:-2]) else 'False'
        print(f"  - {label} ({ok})")

missing = loader.missing_ground_truth(root_path)
if not missing:
    print("No missing ground truth files found. All HSI files have a GT pair.")

print("HSI files without ground truth:")
for name in missing:
    print(f"- {name}")
