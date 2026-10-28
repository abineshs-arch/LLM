"""Creates 8 dummy resumes (5 PDF + 3 TXT) plus 2 broken PDFs (scanned + empty) in resumes/.
All people and companies are fictional. Also saves truth.json (the correct answers) for checking."""
import json
from pathlib import Path

import fitz  # PyMuPDF

BASE = Path(__file__).parent
OUT = BASE / "resumes"
OUT.mkdir(exist_ok=True)

RESUMES = {
    "resume_01.pdf": """Ravi Kumar
Python Developer | Chennai | ravi.kumar@example.com

SUMMARY
Backend developer with 4 years of experience building REST APIs.

EXPERIENCE
Software Engineer, Brightline Tech, Chennai (2022 - present)
- Built REST APIs with Django and FastAPI used by 20,000 users
- Moved services to Docker and AWS, cutting hosting cost by 30%
Junior Developer, Kovai Soft, Coimbatore (2021 - 2022)
- Wrote SQL reports in PostgreSQL

SKILLS
Python, Django, FastAPI, PostgreSQL, Docker, AWS, Git

EDUCATION
B.E. Computer Science, Anna University, 2021""",

    "resume_02.pdf": """Meena Iyer
Data Analyst | Bengaluru

PROFILE
Data analyst with 7 years of experience turning sales data into dashboards.

WORK HISTORY
Senior Data Analyst, Northwind Retail (2020 - present)
- Built Power BI dashboards for 40 stores
- Automated weekly Excel reports with Python
Data Analyst, Lotus Finance (2018 - 2020)
- Wrote SQL queries for credit risk reports

SKILLS
SQL, Excel, Power BI, Python, Tableau

EDUCATION
MBA (Business Analytics), Christ University, 2018
B.Com, Madras University, 2016""",

    "resume_03.pdf": """Arjun S
Frontend Developer (Fresher) | Madurai

ABOUT ME
Recent graduate looking for a frontend role. 6-month internship, no full-time experience yet.

INTERNSHIP
Web Development Intern, PixelCraft Studio (Jan 2025 - Jun 2025)
- Built React components for an e-commerce site
- Wrote a small Node.js API for product search

SKILLS
JavaScript, React, Node.js, HTML, CSS, Git

EDUCATION
B.Sc. Computer Science, Madurai Kamaraj University, 2025""",

    "resume_04.pdf": """Dr. Fatima Khan
Machine Learning Engineer | Hyderabad

SUMMARY
ML engineer with 9 years of experience in NLP and computer vision. Led a team of 6 engineers.

EXPERIENCE
Lead ML Engineer, Vision Labs (2021 - present)
- Led a team of 6 building document-understanding models with PyTorch
- Deployed models with Kubeflow and MLOps pipelines
ML Engineer, DataSpring (2017 - 2021)
- Built NLP models for customer support ticket routing

SKILLS
Python, PyTorch, NLP, Computer Vision, Kubeflow, MLOps, Team Leadership

EDUCATION
PhD Computer Science, IIT Madras, 2017
M.Tech Computer Science, IIT Madras, 2013""",

    "resume_05.pdf": """Karthik Raman
DevOps Engineer | Pune

EXPERIENCE
DevOps Engineer, CloudNest (2023 - present)
- Manage Kubernetes clusters and Terraform infrastructure
- Built Jenkins CI/CD pipelines
QA Engineer, TestPro Solutions (2021 - 2023)
- Manual testing and Selenium automation, bug tracking in JIRA

Total experience: 5 years (2 years QA + 3 years DevOps)

SKILLS
Kubernetes, Terraform, Jenkins, Docker, Selenium, JIRA, Linux

EDUCATION
B.Tech Information Technology, VIT Vellore, 2021""",

    "resume_06.txt": """Priya Natarajan
HR Executive | Chennai

Experience: 3 years in recruitment and employee onboarding.

Recruiter, TalentBridge (2022 - present): hired 120+ candidates for IT roles.
HR Intern, Sunrise Hospital (2021 - 2022): maintained employee records.

Skills: Recruitment, Onboarding, MS Excel, Communication, HRMS

Education: MBA Human Resources, Loyola College, 2021""",

    "resume_07.txt": """Vikram Desai
Sales and Business Development | Mumbai

Worked in B2B software sales for several companies. Consistently beat quarterly targets.
Closed a Rs 2 crore deal with a logistics client.

Skills: Negotiation, CRM (Salesforce), Lead Generation, Presentation

Education: Not mentioned""",

    "resume_08.txt": """Anitha George
Quality Engineer | Coimbatore

Quality engineer with 6 years of experience in automotive manufacturing.

Quality Engineer, Precision Auto Parts (2020 - present)
- Prepared PPAP, PSW and control plans for new parts
- Reduced customer complaints by 25% with root cause analysis (8D)
Quality Inspector, Sri Ganesh Castings (2019 - 2020)

Skills: PPAP, 8D, SPC, FMEA, Minitab, ISO 9001

Education: B.E. Mechanical Engineering, PSG College of Technology, 2019""",
}

# The correct answers, used later to check the model's output
TRUTH = {
    "resume_01.pdf": {"candidate_name": "Ravi Kumar", "years_of_experience": 4,
                      "skills": ["Python", "Django", "FastAPI", "PostgreSQL", "Docker", "AWS", "Git"],
                      "education": ["B.E. Computer Science, Anna University, 2021"]},
    "resume_02.pdf": {"candidate_name": "Meena Iyer", "years_of_experience": 7,
                      "skills": ["SQL", "Excel", "Power BI", "Python", "Tableau"],
                      "education": ["MBA (Business Analytics), Christ University, 2018",
                                    "B.Com, Madras University, 2016"]},
    "resume_03.pdf": {"candidate_name": "Arjun S", "years_of_experience": 0.5,
                      "skills": ["JavaScript", "React", "Node.js", "HTML", "CSS", "Git"],
                      "education": ["B.Sc. Computer Science, Madurai Kamaraj University, 2025"]},
    "resume_04.pdf": {"candidate_name": "Dr. Fatima Khan", "years_of_experience": 9,
                      "skills": ["Python", "PyTorch", "NLP", "Computer Vision", "Kubeflow", "MLOps", "Team Leadership"],
                      "education": ["PhD Computer Science, IIT Madras, 2017", "M.Tech Computer Science, IIT Madras, 2013"]},
    "resume_05.pdf": {"candidate_name": "Karthik Raman", "years_of_experience": 5,
                      "skills": ["Kubernetes", "Terraform", "Jenkins", "Docker", "Selenium", "JIRA", "Linux"],
                      "education": ["B.Tech Information Technology, VIT Vellore, 2021"]},
    "resume_06.txt": {"candidate_name": "Priya Natarajan", "years_of_experience": 3,
                      "skills": ["Recruitment", "Onboarding", "MS Excel", "Communication", "HRMS"],
                      "education": ["MBA Human Resources, Loyola College, 2021"]},
    "resume_07.txt": {"candidate_name": "Vikram Desai", "years_of_experience": None,
                      "skills": ["Negotiation", "CRM (Salesforce)", "Lead Generation", "Presentation"],
                      "education": []},
    "resume_08.txt": {"candidate_name": "Anitha George", "years_of_experience": 6,
                      "skills": ["PPAP", "8D", "SPC", "FMEA", "Minitab", "ISO 9001"],
                      "education": ["B.E. Mechanical Engineering, PSG College of Technology, 2019"]},
}

HEADER = "CONFIDENTIAL - CANDIDATE PROFILE"


def write_pdf(path, text):
    """2-page PDF with a repeated header and 'Page X of 2' footer (to test cleaning)."""
    lines = text.split("\n")
    half = len(lines) // 2
    doc = fitz.open()
    for i, chunk in enumerate([lines[:half], lines[half:]], 1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 40), HEADER, fontsize=8)
        page.insert_textbox(fitz.Rect(50, 70, 545, 780), "\n".join(chunk), fontsize=11)
        page.insert_text((270, 815), f"Page {i} of 2", fontsize=8)
    doc.save(path)
    doc.close()


def write_scanned_pdf(path, text):
    """A PDF that contains only a picture of the text (like a scanned paper) - no real text."""
    tmp = fitz.open()
    page = tmp.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(50, 70, 545, 780), text, fontsize=11)
    pix = page.get_pixmap(dpi=100)
    tmp.close()
    doc = fitz.open()
    doc.new_page(width=595, height=842).insert_image(fitz.Rect(0, 0, 595, 842), pixmap=pix)
    doc.save(path)
    doc.close()


def main():
    for name, text in RESUMES.items():
        if name.endswith(".pdf"):
            write_pdf(OUT / name, text)
        else:
            (OUT / name).write_text(text, encoding="utf-8")
    write_scanned_pdf(OUT / "scanned.pdf", RESUMES["resume_01.pdf"])
    empty = fitz.open()
    empty.new_page()
    empty.save(OUT / "empty.pdf")
    empty.close()
    (BASE / "truth.json").write_text(json.dumps(TRUTH, indent=2), encoding="utf-8")
    print(f"Created {len(RESUMES)} resumes + scanned.pdf + empty.pdf in {OUT.name}/ and truth.json")


if __name__ == "__main__":
    main()