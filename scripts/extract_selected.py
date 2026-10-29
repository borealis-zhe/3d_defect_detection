"""Extract the 12-category original-file subset from the official PCD ZIP.

Only train PCDs, original test PCDs, and corresponding gt TXT files are kept.
This is a file-selection match to the companion's 603-object setup; the later
renderer/reference domain remains independent.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from repro.preprocess import CATEGORIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allowed = set(CATEGORIES)
    with ZipFile(args.zip) as archive:
        files = {}
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            positions = [i for i, part in enumerate(parts) if part in allowed]
            if len(positions) != 1:
                continue
            tail = parts[positions[0]:]
            if len(tail) != 3 or tail[1] not in {"train", "test", "gt"}:
                continue
            category, split, filename = tail
            stem = Path(filename).stem.lower()
            if "cut" in stem or "copy" in stem or "temp" in stem:
                continue
            if split in {"train", "test"} and not filename.lower().endswith(".pcd"):
                continue
            if split == "gt" and not filename.lower().endswith(".txt"):
                continue
            key = (category, split, filename)
            if key in files:
                raise ValueError(f"Duplicate archive member for {key}")
            files[key] = member
        test_stems = {(cat, Path(name).stem) for cat, split, name in files
                      if split == "test"}
        selected = {key: member for key, member in files.items()
                    if key[1] != "gt" or (key[0], Path(key[2]).stem) in test_stems}
        counts = {split: sum(key[1] == split for key in selected)
                  for split in ("train", "test", "gt")}
        print(f"Selected archive files: {counts}", flush=True)
        if counts["train"] != 48 or counts["test"] != 603:
            raise ValueError("Official archive selection does not match the "
                             "expected 48 train / 603 test; inspect before extracting")
        for number, (key, member) in enumerate(sorted(selected.items()), 1):
            target = args.output.joinpath(*key)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            temporary = target.with_name(target.name + ".partial")
            with archive.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary.replace(target)
            if number % 100 == 0 or number == len(selected):
                print(f"Extracted {number}/{len(selected)} files", flush=True)


if __name__ == "__main__":
    main()
