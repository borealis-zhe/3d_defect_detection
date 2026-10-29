"""Known-rotation sanity check of label-blind pose registration."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial.transform import Rotation

from .bundle_ablate import CATEGORIES, ROOT
from .pose_registration import DATA, normalized_raw, register, sample_indices
from .raw_coverage import RAW_ZIP


def main() -> None:
    cases = (("yaw75", (0, 75, 0)), ("roll90", (0, 0, 90)),
             ("tilted", (40, -65, 25)))
    rows = []
    with ZipFile(RAW_ZIP) as archive:
        for category in CATEGORIES:
            path = sorted((DATA / category).glob("train/good/xyz/*/sample.json"))[0]
            meta = json.loads(path.read_text(encoding="utf-8"))
            raw = normalized_raw(archive, meta)
            template = raw[sample_indices(len(raw), 1200, meta["raw_member"])]
            for name, euler in cases:
                known = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
                result = register(template @ known, template)
                net = known @ result["rotation"]
                error = float(np.degrees(np.arccos(np.clip((np.trace(net) - 1) / 2, -1, 1))))
                rows.append({"category": category, "case": name, "angle_error_deg": error,
                             "chamfer_after": result["chamfer_after"],
                             "chamfer_before": result["chamfer_before"]})
    output = ROOT / "outputs/pose_registration/synthetic_rotation_check.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"{len(rows)} known rotations: median angle error "
          f"{np.median([r['angle_error_deg'] for r in rows]):.2f} degrees; "
          f"under 10 degrees {sum(r['angle_error_deg'] < 10 for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
