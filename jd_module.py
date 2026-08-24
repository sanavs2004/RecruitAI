import re
import os
from datetime import datetime
from groq import Groq
from config import Config

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.3-70b-versatile"
STORE_FOLDER = Config.JD_STORE_FOLDER


# ==============================
# Generate Job Description
# ==============================
def generate_job_description(
    role,
    skills,
    experience,
    responsibilities=None,
    education=None,
    soft_skills=None,
    company_name=None,
    company_type=None,
    location=None,
    salary=None,
):
    # Build context blocks only when the field has a value

    company_block = ""
    if company_name or company_type:
        company_block = "COMPANY CONTEXT\n"
        if company_name:
            company_block += f"Company Name: {company_name}\n"
        if company_type:
            type_map = {
                "startup": "Startup — fast-paced, ownership culture, high growth",
                "mnc":     "MNC / Enterprise — structured, global scale, process-driven",
                "product": "Product-based company — own IP, roadmap and innovation driven",
                "service": "Service-based company — client delivery, project-oriented",
            }
            company_block += f"Company Type: {type_map.get(company_type, company_type)}\n"

    work_block = f"WORK DETAILS\nLocation: {location}\n" if location else ""

    salary_block = f"COMPENSATION\nSalary: {salary}\n" if salary else ""

    resp_block = (
        f"KEY RESPONSIBILITIES PROVIDED BY HIRING MANAGER\n{responsibilities}\n"
        "(Expand and enrich these into the JD, keeping every point)"
        if responsibilities else ""
    )

    soft_block = f"Soft Skills Required: {soft_skills}" if soft_skills else ""

    edu_block = f"Minimum Education: {education}" if (education and education != "Any") else ""

    skills_str = ', '.join(skills) if isinstance(skills, list) else skills

    prompt = f"""
You are a Senior HR Director with 20 years of experience writing job descriptions that attract top talent.

Your task: Write a COMPLETE, DETAILED, READY-TO-POST job description using ALL the information provided below.
Tone: formal, corporate, authoritative — Fortune 500 style

IMPORTANT RULES:
- Use EVERY field provided. Do not ignore soft skills, company context, location, or salary.
- If responsibilities are provided, expand EACH ONE into a full rich bullet — never skip or summarize them.
- Match the language to the company type:
    Startup      → equity, ownership, growth, ship fast
    MNC          → scale, stability, global impact, structured
    Product-based → innovation, roadmap ownership, product thinking
    Service-based → client exposure, delivery excellence, variety
- Use hyphen bullets only — no asterisks, no markdown bold, no numbered lists.
- Write a minimum of 600 words.
- Follow the output structure below exactly — every section must appear.

=== INPUT DETAILS ===

Job Title: {role}
Experience Required: {experience} years
Technical Skills: {skills_str}
{edu_block}
{soft_block}
{resp_block}
{company_block}
{work_block}
{salary_block}

=== OUTPUT STRUCTURE (follow exactly, in this order) ===

ABOUT THE ROLE
[4-5 sentences. Describe the role, its strategic importance, the team context, and the impact the hire will make.
Naturally weave in any company context provided.]

KEY RESPONSIBILITIES
[3-5 bullet points starting with a strong action verb.
If responsibilities were given above, expand every single one into a detailed bullet.
Add 2-3 additional responsibilities that logically complement the role.]

REQUIRED QUALIFICATIONS
[Bullet points covering: education requirement, years of experience, every technical skill listed.]

PREFERRED QUALIFICATIONS
[3-4 nice-to-have skills. Include the soft skills provided here.]

WORK DETAILS
[Location as a bullet. If no location was given, skip this section entirely.]

COMPENSATION & BENEFITS
[Show the salary if provided, otherwise write "Competitive, based on experience".
List 3-4 benefits appropriate to the company type — do not skip this section.]


"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7,
    )

    raw_jd = response.choices[0].message.content.strip()

    if not raw_jd:
        raise RuntimeError("Empty response from model")

    cleaned_jd = clean_jd_output(raw_jd)

    os.makedirs(STORE_FOLDER, exist_ok=True)

    with open(os.path.join(STORE_FOLDER, "latest_jd.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned_jd)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_role = re.sub(r'[^\w\s-]', '', role)
    clean_role = re.sub(r'[-\s]+', '_', clean_role)
    filename = f"jd_{clean_role}_{timestamp}.txt"

    with open(os.path.join(STORE_FOLDER, filename), "w", encoding="utf-8") as f:
        f.write(cleaned_jd)

    return cleaned_jd


# ==============================
# Get JD History
# ==============================
def get_all_jds():
    if not os.path.exists(STORE_FOLDER):
        return []

    jds = []
    for file in os.listdir(STORE_FOLDER):
        if file.startswith("jd_") and file.endswith(".txt"):
            path = os.path.join(STORE_FOLDER, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                jds.append({
                    "filename": file,
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "preview": content[:500],
                })
            except UnicodeDecodeError:
                jds.append({
                    "filename": file,
                    "content": "[Binary file — preview not available]",
                    "preview": "[Binary file]",
                })
            except Exception as e:
                print(f"Error reading {file}: {e}")

    jds.sort(key=lambda x: x["filename"], reverse=True)
    return jds


# ==============================
# Clean JD Output
# ==============================
def clean_jd_output(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*',     r'\1', text)
    text = re.sub(r'__(.*?)__',     r'\1', text)

    lines = text.split('\n')
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            cleaned.append('')
            continue
        line = line.replace('•', '-')
        if line.startswith(('*', '•')):
            line = '- ' + line.lstrip('*• ').strip()
        cleaned.append(line)

    # Collapse runs of blank lines to a single blank
    result = []
    prev_blank = False
    for line in cleaned:
        if line == '':
            if not prev_blank:
                result.append(line)
            prev_blank = True
        else:
            prev_blank = False
            result.append(line)

    return '\n'.join(result).strip()