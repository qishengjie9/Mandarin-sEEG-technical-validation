# Dataset interface

## Character-task event table

The within-participant loader searches below each participant directory for:

```text
sub-XX_ses-YY_task-character_desc-words_events.tsv
```

Required columns:

| Column | Meaning |
|---|---|
| `recording_file` | Name of the corresponding `.npy` sEEG trial |
| `start` | Event onset in seconds within the trial |
| `end` | Event offset in seconds within the trial |
| `tone` | Mandarin lexical tone, encoded as 1, 2, 3, or 4 |
| `word` | Pinyin label, preferably including the tone number |

Optional columns:

| Column | Meaning |
|---|---|
| `character` | Chinese character used as the canonical stimulus identifier |
| `original_file_name` | Provenance field retained during dataset conversion |

The corresponding neural file is resolved as:

```text
sub-XX/ses-YY/ieeg/<recording_file>
```

## Neural array

The current release expects a NumPy array with layout:

```text
[time_samples, channels]
```

The dataset-specific sampling rates and known active-channel counts are defined
at the top of `src/mandarin_seeg/preprocessing.py`. Update these mappings when
adapting the code to another dataset.

## Generated training artifacts

Each fold directory contains:

```text
best_model.pt
best_metrics.json
best_predictions.npz
last_model.pt
last_metrics.json
history.json
train_manifest.csv
val_manifest.csv
test_manifest.csv
```

The public figure scripts read only metrics, predictions, and anonymized
participant identifiers. Do not publish manifests if their path fields contain
local filesystem names or information that has not been cleared for release.
