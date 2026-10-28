"""Task 1 report: characters and tokens for every resume, clear errors for broken files."""
from pathlib import Path

from reader import ResumeError, count_tokens, read_resume

BASE = Path(__file__).parent

for path in sorted((BASE / "resumes").iterdir()):
    try:
        text = read_resume(path)
        print(f"{path.name:15} {len(text):5,} characters, {count_tokens(text):4,} tokens")
    except ResumeError as e:
        print(f"{path.name:15} Error - {e}")

print("\nCleaned text of resume_01.pdf (header and page numbers removed):\n")
print(read_resume(BASE / "resumes" / "resume_01.pdf"))