
#!/usr/bin/env python3
"""Inspect ScamSloth's raw datasets without modifying them."""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

RAW = Path("data/raw")
MAX_WARNINGS = 5

SOURCES = {
    "kaggle": {
        "license": "CC0 (verify source page)",
        "synthetic": False,
    },
    "robocall": {
        "license": "Public domain (verify repository)",
        "synthetic": False,
    },
    "bothbosu-scam": {
        "license": "Apache-2.0 (verify source page)",
        "synthetic": True,
    },
    "bothbosu-multi": {
        "license": "Apache-2.0 (verify source page)",
        "synthetic": True,
    },
    "aixblock": {
        "license": "CC BY-NC (verify terms)",
        "synthetic": None,
    },
    "taskmaster": {
        "license": "CC BY 4.0 (verify source)",
        "synthetic": False,
    },
}

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def inspect_json(path):
    with path.open(encoding="utf-8-sig") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        return "object", list(obj.keys())[:15], obj
    if isinstance(obj, list):
        first = obj[0] if obj else None
        keys = list(first.keys())[:15] if isinstance(first, dict) else []
        return f"array ({len(obj)} items)", keys, first
    return type(obj).__name__, [], obj


def normalized_hash(text):
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def check_rows(rows, text_column, label_column=None):
    counts = Counter()
    hashes = Counter()
    empty = 0
    total = 0
    errors = 0

    for row in rows:
        total += 1
        value = row.get(text_column)
        if not isinstance(value, str) or not value.strip():
            empty += 1
        else:
            hashes[normalized_hash(value)] += 1

        if label_column:
            counts[str(row.get(label_column, "<missing>"))] += 1

        if None in row:
            errors += 1

    return {
        "rows": total,
        "empty": empty,
        "duplicates": sum(n - 1 for n in hashes.values()),
        "labels": dict(counts),
        "malformed_csv_rows": errors,
    }


def main():
    print("\nSCAM SLOTH — DATASET VERIFICATION")
    print("=" * 50)

    expected = set(SOURCES)
    present = {
        p.name for p in RAW.iterdir() if p.is_dir()
    } if RAW.exists() else set()

    print("\n1. DOWNLOADS")
    for name in sorted(expected):
        folder = RAW / name
        files = [
            p for p in folder.rglob("*")
            if p.is_file() and ".git" not in p.parts
        ] if folder.exists() else []

        print(f"{name}: {len(files)} files")
        if not files:
            print("  WARNING: Missing or empty folder")

    print("\n2. CSV STRUCTURE, LABELS AND QUALITY")

    paths = [
        ("kaggle", RAW / "kaggle/FullTranscriptData.csv",
         "Content", None),
        ("robocall", RAW / "robocall/metadata.csv",
         "transcript", None),
        ("bothbosu-scam", RAW / "bothbosu-scam/data.csv",
         "dialogue", "label"),
        ("bothbosu-multi-train",
         RAW / "bothbosu-multi/agent_conversation_train.csv",
         "dialogue", "labels"),
        ("bothbosu-multi-test",
         RAW / "bothbosu-multi/agent_conversation_test.csv",
         "dialogue", "labels"),
    ]

    for name, path, text_col, label_col in paths:
        print(f"\n{name}")
        if not path.exists():
            print(f"  MISSING: {path}")
            continue

        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                columns = next(csv.reader(f))

            print(f"  Columns: {columns}")

            if text_col not in columns:
                print(f"  WARNING: Missing text column {text_col}")
                continue
            if label_col and label_col not in columns:
                print(f"  WARNING: Missing label column {label_col}")
                continue

            result = check_rows(
                read_csv(path), text_col, label_col
            )
            for key, value in result.items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    print("\n3. JSON STRUCTURE")

    for name, folder in [
        ("aixblock", RAW / "aixblock"),
        ("taskmaster", RAW / "taskmaster/TM-1-2019"),
    ]:
        print(f"\n{name}")
        if not folder.exists():
            print("  MISSING")
            continue

        paths = sorted(folder.rglob("*.json"))
        print(f"  JSON files: {len(paths)}")

        for path in paths[:3]:
            try:
                kind, keys, example = inspect_json(path)
                print(f"  File: {path.name}")
                print(f"    Structure: {kind}")
                print(f"    Keys: {keys}")
                if isinstance(example, dict):
                    preview = str(example)[:300]
                    print(f"    Preview: {preview}")
            except Exception as e:
                print(f"  ERROR in {path.name}: {e}")

        print("  NOTE: JSON samples inspected, not all rows.")

    print("\n4. LICENSES AND SYNTHETIC STATUS")
    for name, info in SOURCES.items():
        print(
            f"{name}: {info['license']} "
            f"| synthetic={info['synthetic']}"
        )

    print("\n5. ITEMS REQUIRING HUMAN REVIEW")
    print("- Confirm what BothBosu labels 0 and 1 mean.")
    print("- Confirm which robocalls are actual scams.")
    print("- Check AIxBlock JSON format and recording origin.")
    print("- Check Taskmaster spoken versus written dialogues.")
    print("- Verify licenses and redistribution permissions.")
    print("- Inspect cross-source and near-duplicate calls.")
    print("- JSON quality and duplicate counts are not yet checked.")

    print("\nVerification inspection complete.")
    print("This is a diagnostic report, not an approval to freeze.")


if __name__ == "__main__":
    main()
