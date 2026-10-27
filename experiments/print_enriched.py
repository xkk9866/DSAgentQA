import json
from pathlib import Path

d = json.loads(Path("results/metrics_enriched.json").read_text(encoding="utf-8"))
print("decisive", d["decisive"], "tau", d["replug_tau"])
for method, conds in d["test"].items():
    for condition in ("source", "clean", "flip", "real_conflict", "delete", "irrelevant"):
        forced = conds[condition]["forced"]
        selective = conds[condition]["conformal"]["0.1"]
        print(
            f"{method:16} {condition:14} acc={forced['accuracy']:.3f} "
            f"f1={forced['macro_f1']:.3f} harm={selective['harmful_error']:.3f} "
            f"cov={selective['coverage']:.3f} sel={selective['selective_accuracy']:.3f}"
        )
