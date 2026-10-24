#!/usr/bin/env python3
"""Split ScamSloth's processed dataset into train / val / test (step 7).

Input:  data/processed/scam_call_processed_data.jsonl (from build_dataset.py)
Output: data/processed/{train,val,test,extra_not_scam}.jsonl

Rules (see README "How we built this"):
* Synthetic BothBosu rows go to train only, so scores reflect real calls.
* Real scams (robocall, Kaggle) are split ~50/25/25 per source. Look-alike
  calls (same first PREFIX chars) form a group that stays in one split.
* Val and test get one Taskmaster not_scam call per scam, picked as the
  nearest unused length, so length can't give away the label.
* Train gets one Taskmaster call per real scam from the leftovers, nearest
  length available. Short Taskmaster calls run out, so train still leans
  "short means scam"; val/test stay fair and the length-only baseline checks it.
* Unused Taskmaster calls go to extra_not_scam.jsonl for false-alarm tests.

Every run also writes data/processed/manifest.json (step 8): a receipt with
row/label/source counts and a SHA-256 hash per output file, the settings
(seed, caps, prefix, git commit), and the link, version, download date, and
hash of each raw source file, so any reported score traces to exact data.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import subprocess
from datetime import datetime, timezone
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Tuple

SEED = 42
PREFIX = 200
RATIOS = (("train", 0.50), ("val", 0.25), ("test", 0.25))
REAL_SCAM_SOURCES = ("robocall", "kaggle")
NOT_SCAM_POOL = "taskmaster-woz"
INPUT = "data/processed/scam_call_processed_data.jsonl"
OUTPUTS = ("train", "val", "test", "extra_not_scam")
MANIFEST = "data/processed/manifest.json"
DOWNLOADED = "2026-10-22"
# url/version are None where the download left no record; fill them in by hand.
RAW_SOURCES = {
    "kaggle": {"files": ["data/raw/kaggle/FullTranscriptData.csv"],
               "url": "https://www.kaggle.com/datasets/rivalcults/youtube-scam-phone-call-transcripts", "version": None},
    "bothbosu-scam": {"files": ["data/raw/bothbosu-scam/data.csv"],
                      "url": "https://huggingface.co/datasets/BothBosu/scam-dialogue", "version": None},
    "bothbosu-multi": {"files": ["data/raw/bothbosu-multi/agent_conversation_train.csv",
                                 "data/raw/bothbosu-multi/agent_conversation_test.csv"],
                       "url": "https://huggingface.co/datasets/BothBosu/multi-agent-scam-conversation", "version": None},
    "robocall": {"files": ["data/raw/robocall/metadata.csv"],
                 "url": "https://github.com/wspr-ncsu/robocall-audio-dataset",
                 "version": "git 5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272"},
    "taskmaster-woz": {"files": ["data/raw/taskmaster/TM-1-2019/woz-dialogs.json"],
                       "url": "https://github.com/google-research-datasets/Taskmaster",
                       "version": "git d92cb6af3005f1dc09c39e75e7daf4a04905e00b"},
}

Record = Dict[str, Any]


def group_key(item: Record) -> Tuple[str, str]:
    return item["source"], item["text"][:PREFIX]


def split_real_scams(records: List[Record], rng: random.Random) -> Dict[str, List[Record]]:
    """Assign whole look-alike groups to splits, per source, to hit RATIOS."""
    out: Dict[str, List[Record]] = {name: [] for name, _ in RATIOS}
    for source in REAL_SCAM_SOURCES:
        groups: Dict[Tuple[str, str], List[Record]] = defaultdict(list)
        for item in records:
            if item["source"] == source:
                groups[group_key(item)].append(item)
        keys = sorted(groups)
        rng.shuffle(keys)
        total = sum(len(g) for g in groups.values())
        filled = {name: 0 for name, _ in RATIOS}
        for key in keys:
            # Give the group to the split furthest below its target share.
            name = min(RATIOS, key=lambda r: filled[r[0]] / (total * r[1]))[0]
            out[name].extend(groups[key])
            filled[name] += len(groups[key])
    return out


class LengthPool:
    """Unused not_scam calls, searchable by nearest length.

    Taking a call also retires its look-alike group, so near-duplicates never
    land in two different splits.
    """

    def __init__(self, records: List[Record], rng: random.Random):
        shuffled = records[:]
        rng.shuffle(shuffled)  # random tie-break among equal lengths
        self.items = sorted(shuffled, key=lambda r: len(r["text"]))
        self.lengths = [len(r["text"]) for r in self.items]
        self.retired_groups: set = set()

    def take_nearest(self, length: int) -> Record:
        while self.items:
            i = bisect.bisect_left(self.lengths, length)
            candidates = [j for j in (i - 1, i) if 0 <= j < len(self.items)]
            j = min(candidates, key=lambda k: abs(self.lengths[k] - length))
            item = self.items.pop(j)
            self.lengths.pop(j)
            if group_key(item) not in self.retired_groups:
                self.retired_groups.add(group_key(item))
                return item
        raise ValueError("Ran out of not_scam calls to match")

    def remaining(self) -> List[Record]:
        return [r for r in self.items if group_key(r) not in self.retired_groups]


def split(records: List[Record], seed: int = SEED) -> Dict[str, List[Record]]:
    rng = random.Random(seed)
    real_scams = [r for r in records if r["source"] in REAL_SCAM_SOURCES]
    pool = LengthPool([r for r in records if r["source"] == NOT_SCAM_POOL], rng)
    synthetic = [r for r in records if r["synthetic"]]
    known = set(REAL_SCAM_SOURCES) | {NOT_SCAM_POOL}
    unexpected = {r["source"] for r in records if not r["synthetic"] and r["source"] not in known}
    if unexpected:
        raise ValueError(f"No split rule for sources: {sorted(unexpected)}")

    splits = split_real_scams(real_scams, rng)
    # Val and test first, so they get the closest length matches.
    for name in ("val", "test", "train"):
        matches = [pool.take_nearest(len(s["text"]))
                   for s in sorted(splits[name], key=lambda r: len(r["text"]))]
        splits[name].extend(matches)
    splits["train"].extend(synthetic)
    splits["extra_not_scam"] = pool.remaining()
    for rows in splits.values():
        rng.shuffle(rows)
    check(splits)
    return splits


def check(splits: Dict[str, List[Record]]) -> None:
    """Fail loudly on leaks between splits."""
    ids: Dict[str, str] = {}
    groups: Dict[Tuple[str, str], str] = {}
    for name, rows in splits.items():
        for r in rows:
            if r["id"] in ids:
                raise AssertionError(f"{r['id']} in both {ids[r['id']]} and {name}")
            ids[r["id"]] = name
            other = groups.setdefault(group_key(r), name)
            if other != name:
                raise AssertionError(f"look-alike group {group_key(r)[0]} in {other} and {name}")
    for name in ("val", "test"):
        if any(r["synthetic"] for r in splits[name]):
            raise AssertionError(f"synthetic row in {name}")
        labels = Counter(r["label"] for r in splits[name])
        if labels["scam"] != labels["not_scam"]:
            raise AssertionError(f"{name} is not balanced: {dict(labels)}")


def print_summary(splits: Dict[str, List[Record]]) -> None:
    for name, rows in splits.items():
        labels = Counter(r["label"] for r in rows)
        print(f"\n{name}: {len(rows)} rows {dict(labels)}")
        by_source = Counter((r["source"], r["label"]) for r in rows)
        for (source, label), n in sorted(by_source.items()):
            print(f"  {source:22} {label:9} {n}")
        for label in ("scam", "not_scam"):
            lengths = [len(r["text"]) for r in rows if r["label"] == label]
            if lengths:
                print(f"  median length {label:9} {int(median(lengths))}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> Dict[str, Any]:
    def run(*cmd: str) -> str:
        return subprocess.run(cmd, cwd=root, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD") or None,
            # Ignore the manifest itself: it can only be committed after it's written.
            "uncommitted_changes": bool(run("git", "status", "--porcelain", "--",
                                            ".", ":!data/processed/manifest.json"))}


def build_manifest(root: Path, splits: Dict[str, List[Record]], seed: int) -> Dict[str, Any]:
    """Counts, hashes and settings for this run. Call after outputs are written."""
    from build_dataset import CAP, HARD_CAP  # same settings the build used
    outputs = {}
    for name in OUTPUTS:
        rows = splits[name]
        outputs[f"{name}.jsonl"] = {
            "rows": len(rows),
            "labels": dict(sorted(Counter(r["label"] for r in rows).items())),
            "sources": dict(sorted(Counter(r["source"] for r in rows).items())),
            "synthetic_rows": sum(r["synthetic"] for r in rows),
            "sha256": sha256_file((root / INPUT).parent / f"{name}.jsonl"),
        }
    raw = {}
    for source, info in RAW_SOURCES.items():
        raw[source] = {
            "url": info["url"], "version": info["version"], "downloaded": DOWNLOADED,
            "files": {f: sha256_file(root / f) for f in info["files"]},
        }
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {"seed": seed, "cap": CAP, "hard_cap": HARD_CAP,
                     "lookalike_prefix": PREFIX, "split_ratios": dict(RATIOS)},
        "code": git_commit(root),
        "input": {INPUT: sha256_file(root / INPUT)},
        "outputs": outputs,
        "raw_sources": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / INPUT
    with source.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    splits = split(records, args.seed)
    print_summary(splits)
    out_dir = source.parent
    for name in OUTPUTS:
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="") as handle:
            for item in splits[name]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {len(splits[name])} records to {path}")
    manifest = build_manifest(root, splits, args.seed)
    manifest_path = root / MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
