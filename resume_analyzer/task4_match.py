"""Part 2 - Task 2: score all 8 resumes against both jobs (twice each), rank, check consistency."""
import json
from pathlib import Path

from local_llm import MODEL
from matcher import JOBS, PROMPT_VERSION, match

BASE = Path(__file__).parent
RESUMES = BASE / "outputs" / "extract_v1"          # extracted in Part 1
OUT = BASE / "outputs" / PROMPT_VERSION
OUT.mkdir(parents=True, exist_ok=True)
RUNS = 2


def get_result(job_id, stem, resume, run):
    """Cached: if this job/resume/run was already scored, reuse it (lets you stop and resume)."""
    cache = OUT / f"{job_id}__{stem}__run{run}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    result = match(resume, job_id, label=f"{job_id} {stem} run{run}")
    saved = (result["data"].model_dump() | {"success": True}) if result["success"] else result
    cache.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return saved


def main():
    files = sorted(RESUMES.glob("*.json"))
    total = len(files) * len(JOBS) * RUNS
    print(f"Model: {MODEL} | prompt: {PROMPT_VERSION} | temperature 0 | {total} scoring calls\n")
    report = [f"# Job matching results ({MODEL}, prompt {PROMPT_VERSION}, temperature 0)", ""]

    for job_id in JOBS:
        rows = []
        for f in files:
            resume = json.loads(f.read_text(encoding="utf-8"))
            runs = [get_result(job_id, f.stem, resume, r) for r in range(1, RUNS + 1)]
            scores = [r["score"] if r.get("success") else None for r in runs]
            rows.append((resume["candidate_name"], resume["years_of_experience"], scores, runs[0]))

        rows.sort(key=lambda r: -(r[2][0] if r[2][0] is not None else -1))
        title = f"RANKING - {job_id.replace('_', ' ').title()}"
        print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
        print(f"{'#':>2}  {'Candidate':18} {'Yrs':>4} {'Run 1':>6} {'Run 2':>6}  {'Same?':6} Matched skills")
        report += [f"## {title}", "",
                   "| Rank | Candidate | Years | Score run 1 | Score run 2 | Consistent | Matched skills | Missing skills | Reasoning |",
                   "|---|---|---|---|---|---|---|---|---|"]
        for i, (name, yrs, scores, first) in enumerate(rows, 1):
            same = "yes" if scores[0] == scores[1] else "NO"
            matched = ", ".join(first.get("matched_skills", [])) or "-"
            missing = ", ".join(first.get("missing_skills", [])) or "-"
            reason = first.get("reasoning", first.get("message", ""))
            print(f"{i:>2}  {name:18} {str(yrs):>4} {str(scores[0]):>6} {str(scores[1]):>6}  {same:6} {matched}")
            report.append(f"| {i} | {name} | {yrs} | {scores[0]} | {scores[1]} | {same} | {matched} | {missing} | "
                          f"{reason.replace('|', '/')} |")
        report.append("")

    (BASE / "match_results.md").write_text("\n".join(report), encoding="utf-8")
    print("\nSaved match_results.md (full table with missing skills and reasoning)")


if __name__ == "__main__":
    main()