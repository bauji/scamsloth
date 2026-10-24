#!/usr/bin/env python3
"""Build the combined ScamSloth call dataset from the files in data/raw.

Output: data/processed/scam_call_processed_data.jsonl (one JSON object per call).

Pipeline (step 5): load -> clean_text -> cap_text -> normalize_text ->
balance_cut_rates -> validate -> write. Every step removes source artifacts so
the model learns what a call says, not where it came from:
* clean_text: plain ASCII (accents, smart quotes, nbsp, broken chars), drops
  [...] placeholders and speaker labels (caller/receiver/innocent/suspect),
  collapses whitespace (Taskmaster's newlines included).
* cap_text: calls over CAP chars are cut at the first sentence end after CAP;
  with no sentence end before HARD_CAP, hard-cut at HARD_CAP.
* normalize_text: lowercase, no punctuation (Kaggle has none, so punctuation
  would otherwise mean "not Kaggle"). Runs after the cap, which needs .?!
* balance_cut_rates: in BothBosu multi-agent, keep the same number of cut and
  uncut calls per label (seed SEED), so "cut off" doesn't leak the label.
* drop_exact_duplicates (step 6): identical cleaned text keeps only its first
  record. Near-duplicates (same opening) are kept; the split (step 7) keeps
  each look-alike group of real calls in a single split.
The train/val/test split is a later step.
Taskmaster utterances are joined with a newline before cleaning.

Label decisions requiring review before model training or publication:
* Kaggle is labeled scam based on its collection context, not a row label.
* BothBosu uses 1=scam, 0=not_scam, inferred from the supplied type/label
  combinations (ssn/refund/support/reward are 1; delivery/etc. are 0).
* AIxBlock is real, PII-redacted inbound insurance call-center audio
  transcripts, so every record is labeled not_scam (subtype call_center).
  License is CC BY-NC 4.0 with no redistribution: never commit its text.
* Taskmaster WOz is labeled not_scam as background task dialogue, not as a
  verified real-world call. It contains transcribed USER turns and written
  ASSISTANT turns within spoken-dialogue sessions.
* BothBosu is synthetic (Apache-2.0). Kaggle is CC0. Taskmaster's README
  states CC BY 4.0.
* validate_records() fails the build if any row has a missing field, a label
  other than scam/not_scam, an unknown license, or a non-boolean synthetic.

Robocall (NCSU / FTC Project Point of No Entry, public domain) is included:
English-only, deduplicated by transcript, and only calls matching the
IMPERSONATION_PATTERN keywords are kept and labeled scam (subtype
impersonation). Telemarketing and unclear robocalls stay parked.
AIxBlock is parked (not built): every call exceeds the length cap, it is
not_scam only, and its license forbids redistribution. load_aixblock() is kept.
"""

from __future__ import annotations

import argparse
import random
import unicodedata
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


FIELDS = ("id", "text", "label", "subtype", "source", "license", "synthetic")
LABELS = {"scam", "not_scam"}
LICENSES = {
    "kaggle": "CC0",
    "bothbosu-scam": "Apache-2.0",
    "bothbosu-multi-train": "Apache-2.0",
    "bothbosu-multi-test": "Apache-2.0",
    "aixblock": "CC BY-NC 4.0",
    "taskmaster-woz": "CC BY 4.0",
    "robocall": "Public domain",
}
IMPERSONATION_PATTERN = re.compile(
    r"amazon|apple|social security|ssa|irs|police|warrant|arrest|bank|visa|"
    r"mastercard|refund|order placed|suspend",
    re.IGNORECASE,
)
DEFAULT_OUTPUT = "data/processed/scam_call_processed_data.jsonl"
CAP = 2500
HARD_CAP = 2800
SEED = 42
BALANCED_SOURCES = ("bothbosu-multi-train", "bothbosu-multi-test")
ASCII_REPLACEMENTS = str.maketrans({
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2013": "-", "\u2014": "-", "\u00a0": " ",
})
PLACEHOLDER_PATTERN = re.compile(r"\[[^\]]*\]")
SPEAKER_PATTERN = re.compile(r"\b(caller|receiver|innocent|suspect)\s*:", re.IGNORECASE)
SENTENCE_END = re.compile(r"[.?!]")
EXPECTED_INPUTS = {
    "kaggle": 243,
    "bothbosu-scam": 1600,
    "bothbosu-multi-train": 1280,
    "bothbosu-multi-test": 320,
    "aixblock": 1217,
    "taskmaster-woz": 5507,
    "robocall": 1432,
}


def row_id(source: str, local_key: str) -> str:
    digest = hashlib.sha256((source + "\0" + local_key).encode("utf-8")).hexdigest()[:20]
    return source + "-" + digest


def record(source: str, local_key: str, text: str, label: str,
           subtype: Optional[str], license_name: str,
           synthetic: Optional[bool]) -> Dict[str, Any]:
    return dict(zip(FIELDS, (row_id(source, local_key), text, label, subtype,
                             source, license_name, synthetic)))


def valid_text(value: Any) -> bool:
    # The strip is only a missing-value test; retained strings are not stripped.
    return isinstance(value, str) and bool(value.strip())


def csv_rows(path: Path, columns: Tuple[str, ...]) -> Iterator[Tuple[int, Dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(columns).issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected columns {columns}; got {reader.fieldnames}")
        for index, row in enumerate(reader):
            if None in row:
                raise ValueError(f"{path}: malformed CSV row {index + 2}")
            yield index, row


def load_csv(root: Path, relative: str, source: str, text_column: str,
             label_column: Optional[str], subtype_column: Optional[str],
             synthetic: Optional[bool]) -> Tuple[List[Dict[str, Any]], int, int]:
    path = root / relative
    needed = (text_column,) + ((label_column,) if label_column else ()) + (
        (subtype_column,) if subtype_column else ())
    records: List[Dict[str, Any]] = []
    seen = 0
    skipped = 0
    for index, row in csv_rows(path, needed):
        seen += 1
        text = row[text_column]
        if not valid_text(text):
            skipped += 1
            continue
        if label_column:
            raw_label = row[label_column]
            if raw_label not in ("0", "1"):
                raise ValueError(f"{path}: row {index + 2}: unexpected label {raw_label!r}")
            label = "scam" if raw_label == "1" else "not_scam"
        else:
            label = "scam"  # Kaggle collection-level assumption.
        subtype = row[subtype_column] or None if subtype_column else None
        records.append(record(source, f"{relative}:{index}", text, label,
                              subtype, LICENSES[source], synthetic))
    return records, seen, skipped


def load_aixblock(root: Path) -> Tuple[List[Dict[str, Any]], int, int]:
    folder = root / "data/raw/aixblock"
    paths = sorted(folder.rglob("*.json")) if folder.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"No JSON files in {folder}")
    records: List[Dict[str, Any]] = []
    skipped = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict) or "text" not in value:
            raise ValueError(f"{path}: expected an object with a text field")
        text = value["text"]
        if not valid_text(text):
            skipped += 1
            continue
        relative = path.relative_to(root).as_posix()
        records.append(record("aixblock", relative, text, "not_scam", "call_center",
                              LICENSES["aixblock"], False))
    return records, len(paths), skipped


def load_taskmaster(root: Path) -> Tuple[List[Dict[str, Any]], int, int]:
    relative = "data/raw/taskmaster/TM-1-2019/woz-dialogs.json"
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        dialogues = json.load(handle)
    if not isinstance(dialogues, list):
        raise ValueError(f"{path}: expected a dialogue array")
    records: List[Dict[str, Any]] = []
    skipped = 0
    for index, dialogue in enumerate(dialogues):
        if not isinstance(dialogue, dict) or not isinstance(dialogue.get("utterances"), list):
            raise ValueError(f"{path}: dialogue {index} lacks utterances array")
        utterances = dialogue["utterances"]
        texts = [turn.get("text") if isinstance(turn, dict) else None
                 for turn in utterances]
        if not texts or not all(valid_text(text) for text in texts):
            skipped += 1
            continue
        text = "\n".join(texts)
        records.append(record("taskmaster-woz", f"{relative}:{index}", text,
                              "not_scam", None, LICENSES["taskmaster-woz"], False))
    return records, len(dialogues), skipped


def load_robocall(root: Path) -> Tuple[List[Dict[str, Any]], int, int]:
    """Real robocalls: keep English, unique, impersonation-style calls as scam.

    Returns (records, rows_seen, rows_skipped). Skipped covers non-English,
    duplicate transcripts or audio files, empty transcripts, and calls that don't match
    IMPERSONATION_PATTERN (telemarketing / unclear, parked for later).
    """
    relative = "data/raw/robocall/metadata.csv"
    records: List[Dict[str, Any]] = []
    seen_texts = set()
    seen_files = set()  # same audio file can appear twice with slightly different transcripts
    seen = 0
    skipped = 0
    for index, row in csv_rows(root / relative, ("file_name", "language", "transcript")):
        seen += 1
        text = row["transcript"]
        if (row["language"] != "en" or not valid_text(text)
                or text in seen_texts or row["file_name"] in seen_files
                or not IMPERSONATION_PATTERN.search(text)):
            skipped += 1
            continue
        seen_texts.add(text)
        seen_files.add(row["file_name"])
        records.append(record("robocall", row["file_name"], text, "scam",
                              "impersonation", LICENSES["robocall"], False))
    return records, seen, skipped


def clean_text(text: str) -> str:
    """Remove source artifacts: non-ASCII, placeholders, speaker labels, spacing."""
    text = text.translate(ASCII_REPLACEMENTS)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = PLACEHOLDER_PATTERN.sub(" ", text)
    text = SPEAKER_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def cap_text(text: str, cap: int = CAP, hard_cap: int = HARD_CAP) -> Tuple[str, bool]:
    """Cut at the first sentence end after `cap`; hard-cut at `hard_cap`.

    Returns (text, was_cut).
    """
    if len(text) <= cap:
        return text, False
    match = SENTENCE_END.search(text, cap, hard_cap)
    capped = text[:match.end()] if match else text[:hard_cap]
    capped = capped.strip()
    return capped, len(capped) < len(text)


def normalize_text(text: str) -> str:
    """Lowercase and drop punctuation so every source has the same style."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def balance_cut_rates(records: List[Dict[str, Any]], cut_flags: List[bool],
                      seed: int = SEED) -> List[Dict[str, Any]]:
    """In BALANCED_SOURCES, keep equal cut/uncut counts for each label.

    Order of kept rows is preserved. Other sources pass through untouched.
    """
    rng = random.Random(seed)
    keep = [item["source"] not in BALANCED_SOURCES for item in records]
    for source in BALANCED_SOURCES:
        groups: Dict[Tuple[str, bool], List[int]] = {}
        for index, (item, was_cut) in enumerate(zip(records, cut_flags)):
            if item["source"] == source:
                groups.setdefault((item["label"], was_cut), []).append(index)
        for was_cut in (True, False):
            sizes = [len(groups.get((label, was_cut), [])) for label in sorted(LABELS)]
            target = min(sizes)
            for label in sorted(LABELS):
                for index in rng.sample(groups.get((label, was_cut), []), target):
                    keep[index] = True
    return [item for item, kept in zip(records, keep) if kept]


def drop_exact_duplicates(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Keep the first record for each identical cleaned text.

    Fails if identical text carries different labels. Returns (records, dropped).
    """
    first: Dict[str, Dict[str, Any]] = {}
    kept: List[Dict[str, Any]] = []
    for item in records:
        seen = first.get(item["text"])
        if seen is None:
            first[item["text"]] = item
            kept.append(item)
        elif seen["label"] != item["label"]:
            raise ValueError(f"{seen['id']} and {item['id']}: same text, different labels")
    return kept, len(records) - len(kept)


def clean_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Apply clean -> cap -> normalize -> balance. Returns (records, empty_dropped)."""
    cleaned: List[Dict[str, Any]] = []
    cut_flags: List[bool] = []
    empty = 0
    for item in records:
        text, was_cut = cap_text(clean_text(item["text"]))
        text = normalize_text(text)
        if not text:
            empty += 1
            continue
        cleaned.append({**item, "text": text})
        cut_flags.append(was_cut)
    return balance_cut_rates(cleaned, cut_flags), empty


def build(root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[int, int, int]]]:
    specs = (
        ("kaggle", "data/raw/kaggle/FullTranscriptData.csv", "Content", None, None, False),
        ("bothbosu-scam", "data/raw/bothbosu-scam/data.csv", "dialogue", "label", "type", True),
        ("bothbosu-multi-train", "data/raw/bothbosu-multi/agent_conversation_train.csv", "dialogue", "labels", "type", True),
        ("bothbosu-multi-test", "data/raw/bothbosu-multi/agent_conversation_test.csv", "dialogue", "labels", "type", True),
    )
    all_records: List[Dict[str, Any]] = []
    counts: Dict[str, Tuple[int, int, int]] = {}
    for source, relative, text_col, label_col, subtype_col, synthetic in specs:
        rows, seen, skipped = load_csv(root, relative, source, text_col,
                                       label_col, subtype_col, synthetic)
        counts[source] = (seen, len(rows), skipped)
        all_records.extend(rows)
    for source, loader in (("taskmaster-woz", load_taskmaster),
                           ("robocall", load_robocall)):
        rows, seen, skipped = loader(root)
        counts[source] = (seen, len(rows), skipped)
        all_records.extend(rows)
    for source, (seen, included, skipped) in counts.items():
        if seen != EXPECTED_INPUTS[source]:
            raise ValueError(f"{source}: expected {EXPECTED_INPUTS[source]} inputs, found {seen}")
        if included + skipped != seen:
            raise AssertionError(f"{source}: accounting mismatch")
    ids = [item["id"] for item in all_records]
    if len(ids) != len(set(ids)):
        raise AssertionError("Generated record IDs are not unique")
    loaded = len(all_records)
    all_records, empty = clean_records(all_records)
    print(f"Cleaning: {loaded} loaded, {empty} empty after cleaning, "
          f"{loaded - empty - len(all_records)} dropped by cut-rate balancing, "
          f"{len(all_records)} kept")
    all_records, duplicates = drop_exact_duplicates(all_records)
    print(f"Dedupe: {duplicates} exact duplicates dropped, {len(all_records)} kept")
    validate_records(all_records)
    return all_records, counts


def validate_records(records: List[Dict[str, Any]]) -> None:
    """Fail loudly if any row is missing a field or has a bad label/license."""
    problems: List[str] = []
    for item in records:
        rid = item.get("id", "<no id>")
        if tuple(item) != FIELDS:
            problems.append(f"{rid}: fields {list(item)} != {list(FIELDS)}")
            continue
        if not valid_text(item["text"]):
            problems.append(f"{rid}: empty text")
        if item["label"] not in LABELS:
            problems.append(f"{rid}: bad label {item['label']!r}")
        if item["source"] not in LICENSES:
            problems.append(f"{rid}: unknown source {item['source']!r}")
        elif item["license"] != LICENSES[item["source"]]:
            problems.append(f"{rid}: license {item['license']!r} != {LICENSES[item['source']]!r}")
        if not isinstance(item["synthetic"], bool):
            problems.append(f"{rid}: synthetic must be True/False, got {item['synthetic']!r}")
    if problems:
        shown = "\n  ".join(problems[:20])
        raise ValueError(f"{len(problems)} invalid rows, first ones:\n  {shown}")


def print_label_counts(records: List[Dict[str, Any]]) -> None:
    counts: Dict[Tuple[str, str, bool], int] = {}
    for item in records:
        key = (item["source"], item["label"], item["synthetic"])
        counts[key] = counts.get(key, 0) + 1
    print("source / label / synthetic:")
    for (source, label, synthetic), n in sorted(counts.items()):
        print(f"  {source:22} {label:9} synthetic={synthetic!s:5} {n}")


def print_counts(counts: Dict[str, Tuple[int, int, int]]) -> None:
    for source, (seen, included, skipped) in counts.items():
        print(f"{source}: {seen} inputs, {included} included, {skipped} skipped")


def check_output(path: Path, expected: List[Dict[str, Any]]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, wanted in enumerate(expected, 1):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: missing record {index}")
            actual = json.loads(line)
            if list(actual) != list(FIELDS) or actual != wanted:
                raise ValueError(f"{path}: record {index} differs from raw source")
        if handle.readline():
            raise ValueError(f"{path}: extra records after {len(expected)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="ScamSloth project root (default: parent of scripts/)")
    parser.add_argument("--output", type=Path,
                        help=f"JSONL output (default: ROOT/{DEFAULT_OUTPUT})")
    parser.add_argument("--check", action="store_true",
                        help="compare existing JSONL to every raw transcript, without writing")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / DEFAULT_OUTPUT
    records, counts = build(root)
    print_counts(counts)
    print_label_counts(records)
    if args.check:
        check_output(output, records)
        print(f"Validated {len(records)} records in {output}")
        return
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output directory missing: {output.parent}")
    with output.open("x", encoding="utf-8", newline="") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
