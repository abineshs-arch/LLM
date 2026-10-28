"""Part 2 - Task 1: validation + retry with feedback, tested on a messy and an empty resume."""
from extractor import extract_with_retry
from local_llm import MODEL
from reader import MIN_CHARS

MESSY = """SURESH.B   ||  python / SQL guy!!!
   exp:: 2018 -> now  (approx 7 yrs??)    .. our team = 60+ yrs combined experience lol
skills-  python,,sql , excel ;;  powerbi
edu : bsc maths (2017) madras univ
ph: 98xxxxxx        page 1/1"""

EMPTY = "   "


def show(name, result):
    print(f"\n{name}: success={result['success']}, attempts={result['attempts']}")
    if result["success"]:
        print("   ", result["data"].model_dump())
    else:
        print("   FAILURE OBJECT:", {k: v for k, v in result.items()})


def main():
    print(f"Model: {MODEL} | temperature 0 | max 3 attempts | log file: logs/llm_calls.log\n")
    show("Messy resume", extract_with_retry(MESSY, label="messy"))
    print(f"\n(Normally the reader rejects the empty resume: fewer than {MIN_CHARS} characters. "
          "Here we send it to the LLM on purpose to test the retry and the failure object.)")
    show("Empty resume", extract_with_retry(EMPTY, label="empty"))


if __name__ == "__main__":
    main()