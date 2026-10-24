# ScamSloth

Voicemail and scam-call triage agent.

## How we built this

A running log of the engineering decisions behind ScamSloth, and why we made them.

### 1. Dataset (SLOTH-10)

**Goal:** a binary `scam` / `not_scam` call-transcript dataset that teaches the model what scams *say*, not which source a transcript came from.

#### Sources

| Source | Labels | Real or synthetic | License | Used? |
|---|---|---|---|---|
| [Kaggle full transcripts](https://www.kaggle.com/datasets/rivalcults/youtube-scam-phone-call-transcripts) | scam | Real | CC0 | Yes |
| [FTC robocall recordings](https://github.com/wspr-ncsu/robocall-audio-dataset) (Project Point of No Entry) | scam | Real, one-sided (caller only) | Public domain | Yes, impersonation calls only |
| [Taskmaster-1 WOZ dialogs](https://github.com/google-research-datasets/Taskmaster) (Google Research) | not_scam | Real (human-to-human) | CC BY 4.0 | Yes |
| [BothBosu scam dialogues](https://huggingface.co/datasets/BothBosu/scam-dialogue) | scam and not_scam | Synthetic | Apache-2.0 | Yes, train only |
| [BothBosu multi-agent conversations](https://huggingface.co/datasets/BothBosu/multi-agent-scam-conversation) | scam and not_scam | Synthetic | Apache-2.0 | Yes, train only |
| [AIxBlock call-center](https://huggingface.co/datasets/AIxBlock/92k-real-world-call-center-scripts-english) | not_scam | Real | CC BY-NC 4.0 | **No** (see below) |

#### Decisions

| Decision | Why |
|---|---|
| Every record carries `id, text, label, subtype, source, license, synthetic`, and the build validates them | Catches mislabeled rows and unknown licenses early. An earlier draft had AIxBlock labeled `unknown` and missing licenses. |
| Record IDs must be unique, checked at build time | Split, manifest, and eval results all join on `id`. 26 robocall files appeared twice with slightly different transcripts, so we keep the first by `file_name`. This is separate from the content dedupe in the next step. |
| Train on mixed sources, not one | A model trained on a single source learns that source's style. Real data goes in train, val, and test, and synthetic BothBosu data goes **only** in train, so evaluation measures real-world performance. |
| Add FTC robocalls, keeping only impersonation calls | We needed more real scams. A keyword filter (Amazon, Apple, SSA, IRS, police, bank, refund, and similar) kept 567 of 1,432 calls, which a spot check confirmed look like scams. Non-English, telemarketing, and other calls are parked, since telemarketing is spam rather than a scam. |
| Drop AIxBlock | Its transcripts are long (median about 9.6k characters), so the 2,500-character cap would cut 100% of them, and they're all not_scam, which makes "cut off" mean "not scam". The NC license also limits reuse. |
| Spam counts as `not_scam` with a subtype | Spam is unwanted but not fraud. None of the current sources contain spam. |

#### Cleaning

The rule: remove anything that tells you the **source** rather than the **intent**. `scripts/build_dataset.py` runs these in order: clean, then cap, then normalize, then balance.

| Step | Rule | Why |
|---|---|---|
| Clean | Convert to plain ASCII (smart quotes, dashes, accents) | Some sources use fancy characters and others don't, which is a source tell. |
| Clean | Remove placeholder tags like `[person_name]` and speaker labels like `caller:` / `suspect:` | Formatting artifacts that differ by source. |
| Clean | Collapse whitespace and newlines | Spacing reflects the source, not whether a call is a scam. |
| Cap | Over 2,500 characters, cut at the first `.`, `?` or `!` after 2,500. If none by 2,800, hard-cut at 2,800 | Fits model context and cost, and calls end on a full sentence. Kaggle has no punctuation, but all Kaggle calls are under the cap. |
| Normalize | Lowercase and strip all punctuation | Kaggle has no punctuation at all, so punctuation itself would mean "Kaggle, so scam". We considered adding punctuation to Kaggle instead, but that means guessing sentence boundaries and inventing text. Capping happens before this step so the cap can still find sentence ends. |
| Balance | Matched cut rates within BothBosu multi-agent (downsampling) | Keeps "cut off" from leaking the label. See the challenge below. |

#### Challenge: cut-off calls leaking the label

**Problem.** A length cap cuts long calls mid-conversation. If one label gets cut more often than the other, the model can learn "this call got cut off, so it's a scam" instead of learning what scams say. We measured the share of each source and label that the cap would cut:

| Source | Scam cut | Not_scam cut | Leak? |
|---|---|---|---|
| Kaggle, Taskmaster, BothBosu scam, robocall | 0% | 0% | No |
| BothBosu multi-agent | ~69% | ~28% | **Yes** |
| AIxBlock | n/a | 100% | **Yes**, every cut call is not_scam |

**How we got there.**
1. A 2,000-character cap was proposed first, but it would have cut mostly BothBosu scams and left their not_scam calls whole, so we raised it to 2,500.
2. Calls are cut at a sentence end so no call ends mid-word.
3. Even with clean cuts, the content shows whether a call wrapped up ("thanks, bye") or was still going, so the cut rate itself had to match between labels.
4. We first measured lengths before cleaning. Cleaning changes lengths, so we moved cleaning ahead of the cap and re-measured.

**Approaches we rejected.**
- *Trim extra not_scam calls at random points:* they'd end much shorter than cut scams (~2,500), so length becomes the new tell.
- *Trim extra not_scam calls near 2,500:* only ~28% of multi-agent not_scam calls are longer than 2,500, and the cap already cuts those. The rest are too short to cut near 2,500.

**Solution.**
- **Drop AIxBlock.** It can't be balanced because it has no scams.
- **Downsample BothBosu multi-agent** (train and test files separately) so both labels keep the same number of cut and uncut calls, sampled with seed 42. After cleaning, that leaves 382 calls per label in multi-train and 95 per label in multi-test, with a 40% cut rate for both labels. This costs a large share of the multi-agent rows, which is acceptable because they're synthetic and train-only.

#### Dedupe

| Rule | Why |
|---|---|
| Drop exact duplicate texts after cleaning (8,867 to 8,776 records) | Duplicates inflate scores if one copy lands in train and another in test. The build fails if the same text appears with two different labels. |
| Group near-duplicates by source plus the first 200 characters, and keep each group in one split | Exact dedupe misses look-alikes, like the same robocall script with a different name. Keeping a group together stops the model being tested on a call it has effectively already seen. |

#### Split

`scripts/split_dataset.py`, seed 42.

| Split | Rows | Scam / not_scam | Contents |
|---|---|---|---|
| train | 3,278 | 1,639 / 1,639 | 50% of real scams, all synthetic BothBosu data, matching Taskmaster calls |
| val | 364 | 182 / 182 | 25% of real scams, length-matched Taskmaster calls |
| test | 366 | 183 / 183 | 25% of real scams, length-matched Taskmaster calls |
| extra_not_scam | 4,768 | 0 / 4,768 | Leftover Taskmaster calls, not used |

| Decision | Why |
|---|---|
| Real scams (Kaggle, robocall) split 50/25/25 per source, by look-alike group | Real scams are the scarce data, and every split needs both real sources. |
| Synthetic data only in train | Val and test measure real-world performance. The script checks this. |
| Val and test get one Taskmaster call per scam, picked by nearest length | Balanced labels, and length can't give the answer away (median length val 371 vs 371, test 389 vs 388). |
| Train gets the nearest-length leftover Taskmaster calls | Train is balanced, but train still leans "short means scam". The length-only baseline in SLOTH-11 checks whether the model picked that up. |

We considered putting most real scams in train and only a few in val and test. We kept 50/25/25 so the test set is large enough to trust the numbers.

#### Reproducibility (manifest)

`split_dataset.py` writes `data/processed/manifest.json`, which is committed. The jsonl files are not, since anyone can rebuild them. It records:
- the settings (seed, caps, look-alike prefix, split ratios) and the git commit,
- row, label, and source counts for each output, including a check of 0 synthetic rows in val and test,
- a SHA-256 hash for every output file and every raw source file, plus each source's link and version.

If a rebuild gives the same hashes, it's the exact same data.

To rebuild, download the raw sources into `data/raw/`, then run:

```
python3 scripts/build_dataset.py
python3 scripts/split_dataset.py
```

#### Challenge: call length as a shortcut

**Problem.** After the cap, length still differs by source. In the real data, scams are short (robocall median ~330 characters, Kaggle ~766) while normal Taskmaster calls are longer (~901). Robocall and Kaggle are in val and test, so a model could score well just by learning "short means scam." BothBosu scam calls are also longer than its not_scam calls, but that's train-only.

**Solution.**
- **Length-matched sampling for val and test.** The Taskmaster calls in val and test are picked so their length distribution matches the real scams, so length can't give away the answer at evaluation time.
- **Length-only baseline in eval (SLOTH-11).** A classifier that sees only character length. If it scores close to the real model, length is leaking and the results can't be trusted.

#### Known limitations

- **Topic shortcut:** not_scam is mostly Taskmaster bookings (restaurants, rides, tickets), so the model may learn "booking means safe."
- **Style shortcut:** the Kaggle and robocall scams are one-sided (caller only), while Taskmaster is two speakers.
- **Length shortcut:** real scams are shorter than real not_scam calls. This is mitigated in val and test by length-matched sampling and checked with a length-only baseline.
- **Small test set:** 183 real scams in test means scores carry roughly ±4% uncertainty, so small differences between models aren't meaningful.

#### Credits

Taskmaster-1 is by Google Research, used under CC BY 4.0. Kaggle (CC0), robocall (public domain), and BothBosu (Apache-2.0) data are credited in the sources table above.
