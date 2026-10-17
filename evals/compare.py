import argparse
import json
import math
import sys
from collections import Counter

from evals.metric_registry import rule_for

BASELINE_PATH = "baselines/baseline.json"
CANDIDATE_PATH = "baselines/candidate.json"

EXIT_CODE = {"PASS": 0, "FAIL": 1, "REVIEW": 2}


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (
        isinstance(x, float) and math.isnan(x)
    )


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def classify(mid, bv, cv, rule) -> dict:
    """Compare one metric's baseline and candidate values under its rule."""
    row = {"id": mid, "baseline": bv, "candidate": cv, "kind": rule["kind"],
           "direction": rule["direction"], "delta": None, "status": None}

    if rule["kind"] == "info":
        if _is_number(bv) and _is_number(cv):
            row["delta"] = cv - bv
        row["status"] = "info"
        return row

    if _missing(bv):
        row["status"] = "new"
        return row
    if _missing(cv):
        row["status"] = "blocked" if rule["kind"] == "gate" else "missing"
        return row

    if rule.get("bool") or isinstance(bv, bool) or isinstance(cv, bool):
        if bv == cv:
            row["status"] = "flat"
        elif cv and not bv:
            row["status"] = "improved"
        else:
            row["status"] = "blocked" if rule["kind"] == "gate" else "regressed"
        return row

    if not (_is_number(bv) and _is_number(cv)):
        row["status"] = "info"
        return row

    delta = cv - bv
    row["delta"] = delta
    improved = delta > 0 if rule["direction"] == "higher" else delta < 0
    if improved:
        row["status"] = "improved"
        return row

    tolerance = max(rule["tol"], rule["rel_tol"] * abs(bv))
    if abs(delta) <= tolerance:
        row["status"] = "flat"
    else:
        row["status"] = "blocked" if rule["kind"] == "gate" else "regressed"
    return row


def compare(baseline: dict, candidate: dict) -> tuple[str, list[dict]]:
    b, c = baseline.get("metrics", {}), candidate.get("metrics", {})
    rows = [classify(mid, b.get(mid), c.get(mid), rule_for(mid)) for mid in sorted(set(b) | set(c))]

    if any(r["status"] == "blocked" for r in rows):
        verdict = "FAIL"
    elif any(r["status"] in ("regressed", "missing") for r in rows):
        verdict = "REVIEW"
    else:
        verdict = "PASS"
    return verdict, rows


def print_run_differences(bm: dict, cm: dict) -> None:
    changes = []
    for section in ("config", "prompt_hashes"):
        b, c = bm.get(section, {}), cm.get(section, {})
        for key in sorted(set(b) | set(c)):
            if b.get(key) != c.get(key):
                changes.append(f"{section}.{key}: {b.get(key)} -> {c.get(key)}")
    if bm.get("git_sha") != cm.get("git_sha"):
        changes.append(f"git: {bm.get('git_sha')} -> {cm.get('git_sha')}")
    if cm.get("git_dirty"):
        changes.append("candidate has uncommitted changes")

    print("changed   : " + ("; ".join(changes) if changes else "nothing recorded (same settings, prompts, commit)"))
    if cm.get("failed_evals"):
        print(f"WARNING   : candidate is INCOMPLETE, failed evals -> {cm['failed_evals']}")
    if bm.get("failed_evals"):
        print(f"WARNING   : baseline is INCOMPLETE, failed evals -> {bm['failed_evals']}")


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.4g}"
    return "-" if v is None else str(v)


def print_report(verdict: str, rows: list[dict], show_all: bool = False) -> None:
    order = {"blocked": 0, "regressed": 1, "missing": 2, "new": 3, "improved": 4, "flat": 5, "info": 6}
    shown = [r for r in rows if show_all or r["kind"] != "info"]
    shown.sort(key=lambda r: (order.get(r["status"], 9), r["id"]))

    print("=" * 104)
    print(f"{'metric':<56} {'baseline':>10} {'candidate':>10} {'delta':>10}  status")
    print("-" * 104)
    for r in shown:
        d = r["delta"]
        delta = f"{d:+.4g}" if isinstance(d, (int, float)) and not isinstance(d, bool) else ""
        marker = "  <<<" if r["status"] in ("blocked", "regressed", "missing") else ""
        print(f"{r['id']:<56} {_fmt(r['baseline']):>10} {_fmt(r['candidate']):>10} "
              f"{delta:>10}  {r['status']}{marker}")

    counts = Counter(r["status"] for r in rows)
    print("-" * 104)
    print("  ".join(f"{k}={counts[k]}" for k in order if counts[k]))
    if not show_all and counts["info"]:
        print(f"({counts['info']} info metrics hidden; pass --all to show them)")
    print("=" * 104)
    print("VERDICT: " + {
        "PASS": "PASS   -- no gate blocked, no guardrail regressed. Safe to promote.",
        "REVIEW": "REVIEW -- a guardrail regressed or a metric is missing. A person decides.",
        "FAIL": "FAIL   -- a gate regressed or is missing. Blocked.",
    }[verdict])
    print("=" * 104)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two eval snapshots and return a verdict.")
    parser.add_argument("--baseline", default=BASELINE_PATH)
    parser.add_argument("--candidate", default=CANDIDATE_PATH)
    parser.add_argument("--all", action="store_true", help="also show info metrics")
    args = parser.parse_args()

    baseline, candidate = load(args.baseline), load(args.candidate)
    bm, cm = baseline.get("metadata", {}), candidate.get("metadata", {})
    print(f"baseline  : {bm.get('label') or args.baseline}   ({bm.get('created_at', '?')}, git {bm.get('git_sha', '?')})")
    print(f"candidate : {cm.get('label') or args.candidate}   ({cm.get('created_at', '?')}, git {cm.get('git_sha', '?')})")
    print_run_differences(bm, cm)

    verdict, rows = compare(baseline, candidate)
    print_report(verdict, rows, show_all=args.all)
    sys.exit(EXIT_CODE[verdict])


if __name__ == "__main__":
    main()