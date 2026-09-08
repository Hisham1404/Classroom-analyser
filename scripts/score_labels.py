"""Score the filled-in labelling sheets.

    python scripts/score_labels.py

Reads every `eval/labels/*.md`, compares the Truth column against the classifier, and
reports accuracy. Rows left blank are excluded — never counted as correct.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config                                       # noqa: E402
from src.attribution import Role                             # noqa: E402
from src.evaluation.labelling import parse_sheet, score_labels  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default=str(config.EVAL_DIR / "labels"))
    args = ap.parse_args()

    sheets = sorted(Path(args.labels_dir).glob("*.md"))
    if not sheets:
        print(f"no sheets in {args.labels_dir} — run make_labelling_sheet.py first",
              file=sys.stderr)
        return 1

    all_rows = []
    print(f"{'SESSION':<34}{'rows':>6}{'labelled':>10}{'accuracy':>10}")
    print("-" * 60)
    for sheet in sheets:
        rows = parse_sheet(sheet.read_text(encoding="utf-8"))
        all_rows.extend(rows)
        s = score_labels(rows, confidence_floor=config.ROLE_CONF_FLOOR)
        acc = "—" if s.accuracy is None else f"{s.accuracy:.1%}"
        print(f"{sheet.stem[:32]:<34}{s.total:>6}{s.labelled:>10}{acc:>10}")

    overall = score_labels(all_rows, confidence_floor=config.ROLE_CONF_FLOOR)
    print("-" * 60)
    if overall.labelled == 0:
        print("\nNothing labelled yet — accuracy is unknown, which is the honest answer.")
        print("Open the sheets in eval/labels/, play the wavs beside them, fill in Truth.")
        return 0

    print(f"\nOVERALL   accuracy {overall.accuracy:.1%}  "
          f"({overall.correct}/{overall.labelled} labelled rows)")
    if overall.confident_accuracy is not None:
        print(f"          accuracy on confident rows (>= {config.ROLE_CONF_FLOOR}): "
              f"{overall.confident_accuracy:.1%}")
    for role in Role:
        if role in overall.recall:
            print(f"          recall[{role.value}] {overall.recall[role]:.1%}")
    if overall.confusion:
        print("\n  truth -> predicted")
        for (truth, pred), n in sorted(overall.confusion.items(), key=lambda kv: -kv[1]):
            mark = "  " if truth is pred else " x"
            print(f"   {mark} {truth.value:<8} -> {pred.value:<8} {n}")

    out = Path(args.labels_dir) / "score.json"
    out.write_text(json.dumps(overall.to_dict(), indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
