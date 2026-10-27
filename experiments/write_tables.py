"""Write results/tables.md from results/metrics.json. Does not recompute metrics."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
conditions = ("source", "clean", "flip", "real_conflict", "delete", "irrelevant")
methods = list(data["test"])
lines = [
    "# Test results",
    "",
    "Split seed 42. Train 250 is unused. Validation 250 selects thresholds. Test 500 is reported once.",
    f"Mental-health items in the full labeled set: {data['n_mental_health_all']}. In the test split: {data['n_mental_health_test']}.",
    "",
    "Evidence unit: each paper contributes one block, the full RESULTS section (up to 1600 characters), not a 600-character cut.",
    f"Encoder: {data['notes']['encoder']}.",
    "",
    "## Forced answer",
    "",
    "| method | " + " | ".join(f"{name} acc | {name} f1" for name in conditions) + " |",
    "|---|" + "---|" * (2 * len(conditions)),
]
for method in methods:
    cells = []
    for condition in conditions:
        forced = data["test"][method]["conditions"][condition]["forced"]
        cells.append(f"{forced['accuracy']:.3f} | {forced['macro_f1']:.3f}")
    lines.append(f"| {method} | " + " | ".join(cells) + " |")

lines += [
    "",
    "## Selective decision",
    "",
    "Thresholds were chosen on validation by clean success minus flip harmful-error. Abstention is not a success on the clean set.",
    "",
    "| method | clean harm | clean cov | clean sel | flip harm | flip cov | flip sel | real_conflict harm | real_conflict cov | real_conflict sel | delete harm | delete cov | delete sel |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for method in methods:
    cells = []
    for condition in ("clean", "flip", "real_conflict", "delete"):
        selective = data["test"][method]["conditions"][condition]["selective"]
        cells.append(
            f"{selective['harmful_error']:.3f} | {selective['coverage']:.3f} | {selective['selective_accuracy']:.3f}"
        )
    lines.append(f"| {method} | " + " | ".join(cells) + " |")

lines += [
    "",
    "## Calibration on clean",
    "",
    "| method | ECE | Brier | NLL | AURC |",
    "|---|---|---|---|---|",
]
for method in methods:
    forced = data["test"][method]["conditions"]["clean"]["forced"]
    lines.append(
        f"| {method} | {forced['ece']:.3f} | {forced['brier']:.3f} | {forced['nll']:.3f} | {forced['aurc']:.3f} |"
    )

lines += [
    "",
    "## Citation on clean",
    "",
    "| method | precision | recall | attribution correctness |",
    "|---|---|---|---|",
]
for method in methods:
    cited = data["test"][method]["citation_clean"]
    lines.append(
        f"| {method} | {cited['citation_precision']:.3f} | {cited['citation_recall']:.3f} | {cited['attribution_correctness']:.3f} |"
    )

lines += [
    "",
    "## Chosen validation parameters",
    "",
    "| method | utility | threshold | conflict slack | margin | decisive |",
    "|---|---|---|---|---|---|",
]
for method in methods:
    params = data["params"][method]
    decisive = params.get("decisive")
    decisive_text = f"{decisive:.2f}" if isinstance(decisive, float) else "-"
    lines.append(
        f"| {method} | {params['utility']:.3f} | {params['threshold']:.2f} | "
        f"{params['conflict_slack']:.2f} | {params['margin']:.2f} | {decisive_text} |"
    )

lines += [
    "",
    "## Flip harm relative to CAER",
    "",
    "Difference is harm(method) − harm(CAER) on the test flip set. A negative value means that method has fewer harmful answers. Intervals are percentile bootstrap, 2000 resamples.",
    "",
    "| method | delta | low | high |",
    "|---|---|---|---|",
]
for method, stats in data["harm_reduction_bootstrap"]["flip"].items():
    lines.append(f"| {method} | {stats['mean']:.3f} | {stats['low']:.3f} | {stats['high']:.3f} |")

out = ROOT / "results" / "tables.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out}")
for name in ("kg_filter", "typed_yager", "caer", "mean"):
    block = data["relation_probe"][name]
    print(
        name,
        "forced",
        round(block["forced"]["accuracy"], 3),
        "cov",
        round(block["selective"]["coverage"], 3),
        "sel",
        round(block["selective"]["selective_accuracy"], 3),
    )
