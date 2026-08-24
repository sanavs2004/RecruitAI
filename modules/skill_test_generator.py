import os
import json
import uuid
from datetime import datetime
from groq import Groq

from config import Config
client = Groq(api_key=Config.GROQ_API_KEY)
MODEL  = "llama-3.3-70b-versatile"


def generate_skill_test(candidate: dict, job: dict) -> dict:
    """
    Generate MCQ skill test for HOLD candidates using Groq.
    Returns a test dict with questions, options, answers.
    """
    skills       = candidate.get('extracted_skills', [])[:5]
    job_title    = job.get('title', 'the role')
    missing      = [m['skill'] if isinstance(m, dict) else m
                    for m in candidate.get('missing_skills', [])[:3]]

    prompt = f"""
You are a technical interviewer. Generate exactly 5 multiple choice questions to test a candidate for the role of {job_title}.

Focus on these skills: {', '.join(skills + missing)}

Return ONLY valid JSON in this exact format, nothing else:
{{
  "questions": [
    {{
      "id": 1,
      "question": "question text here",
      "options": {{
        "A": "option A text",
        "B": "option B text",
        "C": "option C text",
        "D": "option D text"
      }},
      "correct": "A",
      "skill": "skill being tested"
    }}
  ]
}}

Rules:
- Exactly 5 questions
- Each question tests a specific technical skill
- One clearly correct answer per question
- Other options must be plausible but wrong
- No trick questions
- Difficulty: intermediate level
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.5,
        )
        raw  = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw  = raw.replace('```json', '').replace('```', '').strip()
        data = json.loads(raw)

        test_id = str(uuid.uuid4())
        return {
            'test_id'     : test_id,
            'candidate_id': candidate.get('candidate_id'),
            'candidate'   : candidate.get('filename'),
            'email'       : candidate.get('email'),
            'job_title'   : job_title,
            'skills_tested': skills + missing,
            'questions'   : data['questions'],
            'total'       : len(data['questions']),
            'created_at'  : datetime.now().isoformat(),
            'status'      : 'pending',
            'score'       : None,
            'passed'      : None,
            'pass_mark'   : 60,  # 60% to pass = 3/5 correct
        }

    except Exception as e:
        print(f"❌ Test generation failed: {e}")
        return None