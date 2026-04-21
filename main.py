import io
import json
import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

# Explicitly load .env from the same directory as this file
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

_api_key = os.environ.get("GEMINI_API_KEY", "")
print(f"[startup] KEY loaded: {bool(_api_key)}")
print(f"[startup] KEY prefix: {_api_key[:8]}...")

genai.configure(api_key=_api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPT = """
Analyze this course syllabus and extract the grading policy.
Return ONLY a valid JSON object — no markdown, no explanation.

Schema:
{
  "course_name": "e.g. CS 225",
  "course_subtitle": "e.g. Data Structures",
  "components": [
    {
      "id": "unique_snake_case_id",
      "label": "Display Name",
      "weight": 15,
      "group": "Group Name for UI card",
      "group_icon": "one emoji",
      "group_color": "ic-blue",
      "is_final": false,
      "is_replaceable": false
    }
  ],
  "replacement_policy": {
    "enabled": false,
    "description": "e.g. Final replaces lowest midterm if higher"
  },
  "grade_thresholds": {
    "A": 93, "A-": 90, "B+": 87, "B": 83, "B-": 80, "C+": 77, "C": 73
  },
  "notes": "any important grading notes"
}

Rules:
1. All component weights must sum to exactly 100.
2. Create a separate entry for each individual exam (e.g. Midterm 1, Midterm 2).
3. is_final: true ONLY for the final exam component.
4. is_replaceable: true for midterms/exams that can be replaced by the final.
5. group_color must be one of: ic-purple, ic-teal, ic-blue, ic-orange, ic-red, ic-gold.
6. Use fitting emojis for group_icon: 📋 participation, 📚 homework, 🧪 lab, 📝 exams, 🏆 final, 💻 coding, 🎯 project, 📊 quizzes.
7. ALWAYS populate grade_thresholds. If the syllabus uses a curve or doesn't state fixed cutoffs, still return the standard defaults: A:93, A-:90, B+:87, B:83, B-:80, C+:77, C:73. Never leave grade_thresholds empty or null.
"""


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(None),
    notes: str = Form(""),
):
    if not (file and file.filename) and not notes.strip():
        raise HTTPException(400, "Provide a syllabus file or notes.")

    prompt_parts = [PROMPT]

    # Extract text from file instead of using file upload API
    if file and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in (".pdf", ".txt"):
            raise HTTPException(400, "Only PDF and TXT files are supported.")

        content = await file.read()

        if suffix == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            ).strip()
            if not text:
                raise HTTPException(422, "Could not extract text from PDF.")
        else:
            text = content.decode("utf-8", errors="ignore")

        prompt_parts.append(f"\nSyllabus content:\n{text}")

    if notes.strip():
        prompt_parts.append(f"\nUser-provided notes:\n{notes}")

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt_parts)
        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0].strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Gemini returned unparseable JSON: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}
