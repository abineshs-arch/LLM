"""Task 2: run the extraction prompt on all 8 resumes, save the JSON, count errors for 3 resumes."""
import json
import time
from pathlib import Path

from extractor import MODEL, PROMPT_VERSION, ExtractionError, extract
from reader import ResumeError, read_resume

BASE = Path(__file__).parent
OUT = BASE / "outputs" / PROMPT_VERSION
OUT.mkdir(parents=True, exist_ok=True)
CHECK = ["resume_01.pdf", "resume_03.pdf", "resume_07.txt"]   # the 3 resumes we verify


def norm(s):
    return " ".join(str(s).lower().replace(".", "").split())


def count_errors(pred, truth):
    """Compare one extraction with the correct answer. Returns a list of error descriptions."""
    errors = []
    if norm(pred["candidate_name"] or "") != norm(truth["candidate_name"]):
        errors.append(f"name: got {pred['candidate_name']!r}")
    p, t = pred["years_of_experience"], truth["years_of_experience"]
    if (p is None) != (t is None) or (p is not None and abs(p - t) > 0.01):
        errors.append(f"years_of_experience: got {p}, expected {t}")
    got, want = {norm(s) for s in pred["skills"]}, {norm(s) for s in truth["skills"]}
    errors += [f"missing skill: {s}" for s in sorted(want - got)]
    errors += [f"extra skill (not in resume?): {s}" for s in sorted(got - want)]
    if len(pred["education"]) != len(truth["education"]):
        errors.append(f"education: got {len(pred['education'])} entries, expected {len(truth['education'])}")
    return errors


def main():
    truth = json.loads((BASE / "truth.json").read_text(encoding="utf-8"))
    print(f"Model: {MODEL} | prompt: {PROMPT_VERSION} | JSON mode + temperature 0\n")
    results = {}
    for path in sorted((BASE / "resumes").iterdir()):
        try:
            text = read_resume(path)
        except ResumeError as e:
            print(f"{path.name:15} skipped - {e}")
            continue
        start = time.time()
        try:
            data, _ = extract(text)
        except ExtractionError as e:
            print(f"{path.name:15} FAILED - {e}")
            continue
        results[path.name] = data.model_dump()
        (OUT / f"{path.stem}.json").write_text(json.dumps(results[path.name], indent=2), encoding="utf-8")
        print(f"{path.name:15} valid JSON ({time.time() - start:.0f} s): {data.candidate_name}, "
              f"{data.years_of_experience} yrs, {len(data.skills)} skills, {len(data.education)} degrees")

    print(f"\nValid JSON for {len(results)}/8 resumes, saved in outputs/{PROMPT_VERSION}/")
    print("\n" + "=" * 70 + "\nERROR CHECK FOR 3 RESUMES (against truth.json)\n" + "=" * 70)
    total = 0
    for name in CHECK:
        if name not in results:
            print(f"{name}: no output")
            continue
        errs = count_errors(results[name], truth[name])
        total += len(errs)
        print(f"\n{name}: {len(errs)} error(s)")
        for e in errs:
            print(f"   - {e}")
        print(f"   strengths: {results[name]['strengths']}")
        print(f"   gaps:      {results[name]['gaps']}")
    print(f"\nTOTAL ERRORS in 3 resumes: {total}")


if __name__ == "__main__":
    main()