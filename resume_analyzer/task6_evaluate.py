"""Part 4 - Task 1: evaluate extraction (golden set) for prompt v1 and v2, and score consistency."""
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
TRUTH = json.loads((BASE / "truth.json").read_text(encoding="utf-8"))     # golden set: 8 resumes


def words(s):
    return set(re.findall(r"[a-z0-9+#]+", s.lower()))


def same_skill(a, b):
    """'Power BI' = 'PowerBI', 'MS Excel' = 'Excel', 'CRM (Salesforce)' = 'CRM'."""
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return False
    if "".join(sorted(wa)) == "".join(sorted(wb)) or "".join(re.findall(r"[a-z0-9+#]", a.lower())) == \
            "".join(re.findall(r"[a-z0-9+#]", b.lower())):
        return True
    return wa <= wb or wb <= wa


def match_count(pred, truth):
    """Greedy one-to-one matching between predicted and true skills."""
    used, hits = set(), 0
    for p in pred:
        for i, t in enumerate(truth):
            if i not in used and same_skill(p, t):
                used.add(i)
                hits += 1
                break
    return hits


def norm_name(s):
    return " ".join((s or "").lower().replace(".", " ").split())


def evaluate(version):
    folder = BASE / "outputs" / version
    tp = n_pred = n_true = 0
    field_ok = field_total = 0
    rows = []
    for name, truth in TRUTH.items():
        f = folder / f"{Path(name).stem}.json"
        pred = json.loads(f.read_text(encoding="utf-8")) if f.exists() else \
            {"candidate_name": None, "skills": [], "years_of_experience": None, "education": []}
        hits = match_count(pred["skills"], truth["skills"])
        tp, n_pred, n_true = tp + hits, n_pred + len(pred["skills"]), n_true + len(truth["skills"])
        p, t = pred["years_of_experience"], truth["years_of_experience"]
        checks = {
            "name": norm_name(pred["candidate_name"]) == norm_name(truth["candidate_name"]),
            "years": (p is None and t is None) or (p is not None and t is not None and abs(p - t) < 0.01),
            "education": len(pred["education"]) == len(truth["education"]),
        }
        field_ok += sum(checks.values())
        field_total += len(checks)
        wrong = [k for k, ok in checks.items() if not ok]
        rows.append((name, hits, len(pred["skills"]), len(truth["skills"]), wrong))
    precision = tp / n_pred if n_pred else 0
    recall = tp / n_true if n_true else 0
    return precision, recall, field_ok / field_total, rows


def score_consistency():
    folder = BASE / "outputs" / "match_v1"
    diffs = []
    for run1 in folder.glob("*__run1.json"):
        run2 = folder / run1.name.replace("__run1", "__run2")
        a, b = json.loads(run1.read_text()), json.loads(run2.read_text())
        if "score" in a and "score" in b:
            diffs.append(abs(a["score"] - b["score"]))
    return diffs


def main():
    results = {}
    for version in ["extract_v1", "extract_v2"]:
        if not (BASE / "outputs" / version).exists():
            print(f"{version}: no outputs yet - run task2_extract.py with this prompt first")
            continue
        precision, recall, accuracy, rows = evaluate(version)
        results[version] = (precision, recall, accuracy)
        print("=" * 70 + f"\n{version}\n" + "=" * 70)
        for name, hits, n_pred, n_true, wrong in rows:
            print(f"  {name:15} skills matched {hits}/{n_true} (predicted {n_pred})"
                  + (f" | wrong: {', '.join(wrong)}" if wrong else ""))
        print(f"  -> skill precision {precision:.2f} | recall {recall:.2f} | "
              f"field accuracy (name, years, education) {accuracy:.2f}\n")

    if len(results) == 2:
        (p1, r1, a1), (p2, r2, a2) = results["extract_v1"], results["extract_v2"]
        print(f"Skill extraction precision {p1:.2f}, recall {r1:.2f}, accuracy {a1:.2f}"
              f"  ->  after prompt fix {p2:.2f}, {r2:.2f}, {a2:.2f}")

    diffs = score_consistency()
    if diffs:
        print(f"\nScore consistency: {len(diffs)} resume/job pairs scored twice at temperature 0 -> "
              f"average difference {sum(diffs) / len(diffs):.1f} points (max {max(diffs)})")


if __name__ == "__main__":
    main()