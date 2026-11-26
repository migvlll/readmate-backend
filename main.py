import os
import json
import time
import tempfile
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore
from pypdf import PdfReader
from docx import Document

from assessment_engine import (
    transcribe_audio,
    compute_word_levenshtein,
    compute_wpm,
    evaluate_comprehension,
    classify_reading_level
)

if not firebase_admin._apps:
    key_path = os.getenv("FIREBASE_KEY_PATH", "serviceAccountKey.json")
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

app = FastAPI(
    title="ReadMate Automated Reading Assessment API",
    description="Backend service for Whisper ASR, Levenshtein/WER, NLP, Random Forest, and Firestore.",
    version="1.0.0"
)

# Allow requests from Base44 web preview, Flutter mobile app, and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "online", "service": "ReadMate Assessment Backend"}


@app.post("/api/assess")
async def assess_student_reading(
    audio: UploadFile = File(...),
    student_id: str = Form(...),
    student_name: str = Form(...),
    class_id: str = Form(...),
    material_id: str = Form(...),
    material_title: str = Form(...),
    reference_text: str = Form(...),
    language: str = Form("en"),
    duration_seconds: float = Form(...),
    student_answers_json: str = Form("[]"),
    correct_answers_json: str = Form("[]")
):
    """
    Complete Reading Assessment Pipeline:
    1. Temporarily buffers uploaded audio
    2. Transcribes speech via Whisper ASR
    3. Immediately deletes audio file (Zero-Retention Audio Policy)
    4. Computes Levenshtein Distance, WER, Reading Accuracy, and WPM
    5. Evaluates Comprehension Quiz Score
    6. Classifies Reading Level via Random Forest
    7. Saves structured results to Firebase Firestore
    """
    start_time = time.time()
    temp_audio_path = None

    def parse_answers(raw_value: str) -> list[str]:
        cleaned = raw_value.strip()
        if not cleaned:
            return []
        if cleaned.startswith("["):
            return json.loads(cleaned)
        return [item.strip() for item in cleaned.split(",") if item.strip()]

    try:
        student_answers = parse_answers(student_answers_json)
        correct_answers = parse_answers(correct_answers_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for quiz answers.")

    try:
        # 1. Save audio temporarily in memory/temp disk for Whisper processing
        suffix = os.path.splitext(audio.filename or "recording.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            temp_audio_path = tmp_file.name
            content = await audio.read()
            tmp_file.write(content)

        # 2. Speech-to-Text via Whisper ASR
        transcribed_text = transcribe_audio(temp_audio_path, language=language)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech recognition failed: {str(e)}")

    finally:
        # 3. ZERO-RETENTION AUDIO POLICY: Always delete raw student voice recording
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    # 4. Text Comparison (Levenshtein Distance & WER) and Fluency (WPM)
    text_metrics = compute_word_levenshtein(reference_text, transcribed_text)
    wpm = compute_wpm(text_metrics["words_read"], duration_seconds)

    # 5. Comprehension Evaluation
    comp_metrics = evaluate_comprehension(student_answers, correct_answers)

    # 6. Random Forest Reading Proficiency Classification
    ml_result = classify_reading_level(
        reading_accuracy=text_metrics["reading_accuracy"],
        wpm=wpm,
        comprehension_score=comp_metrics["comprehension_score"]
    )

    # Compute overall weighted score (50% Accuracy, 30% Comprehension, 20% Fluency benchmarked at 120 WPM)
    fluency_pct = min(100.0, round((wpm / 120.0) * 100.0, 2))
    overall_score = round(
        (0.5 * text_metrics["reading_accuracy"]) +
        (0.3 * comp_metrics["comprehension_score"]) +
        (0.2 * fluency_pct),
        2
    )

    processing_time_sec = round(time.time() - start_time, 2)

    # 7. Store Assessment Record in Firebase Firestore
    assessment_record = {
        "student_id": student_id,
        "student_name": student_name,
        "class_id": class_id,
        "material_id": material_id,
        "material_title": material_title,
        "transcribed_text": transcribed_text,
        "total_words": text_metrics["total_words"],
        "words_read": text_metrics["words_read"],
        "duration_seconds": duration_seconds,
        "reading_accuracy": text_metrics["reading_accuracy"],
        "wer": text_metrics["wer"],
        "wpm": wpm,
        "fluency_percentage": fluency_pct,
        "comprehension_score": comp_metrics["comprehension_score"],
        "quiz_correct": comp_metrics["correct_count"],
        "quiz_total": comp_metrics["total_questions"],
        "overall_score": overall_score,
        "reading_level": ml_result["reading_level"],
        "model_confidence": ml_result["confidence"],
        "error_breakdown": {
            "substitutions": text_metrics["substitutions"],
            "deletions": text_metrics["deletions"],
            "insertions": text_metrics["insertions"]
        },
        "error_list": text_metrics["error_list"],
        "processing_time_seconds": processing_time_sec,
        "audio_retained": False,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    doc_ref = db.collection("assessments").add(assessment_record)
    assessment_id = doc_ref[1].id

    # Update student's latest proficiency level and status in Firestore
    status = "Needs support" if ml_result["reading_level"] in ["Beginner", "Developing"] else "Active"
    db.collection("students").document(student_id).set({
        "current_level": ml_result["reading_level"],
        "last_score": overall_score,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }, merge=True)

    return {
        "assessment_id": assessment_id,
        "data": assessment_record
    }


@app.post("/api/materials/extract-text")
async def extract_material_text(file: UploadFile = File(...)):
    """
    Supports the Teacher Web App 'Upload File' wizard (/teacher/materials/new)
    by extracting plain text and word counts from .pdf, .docx, or .txt files.
    """
    filename = (file.filename or "").lower()
    extracted_text = ""

    try:
        if filename.endswith(".txt"):
            raw_bytes = await file.read()
            extracted_text = raw_bytes.decode("utf-8", errors="ignore").strip()

        elif filename.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            try:
                reader = PdfReader(tmp_path)
                pages = [page.extract_text() or "" for page in reader.pages]
                extracted_text = "\n".join(pages).strip()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        elif filename.endswith(".docx"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            try:
                doc = Document(tmp_path)
                extracted_text = "\n".join([p.text for p in doc.paragraphs]).strip()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload .pdf, .docx, or .txt.")

        word_count = len(extracted_text.split())
        est_minutes = max(1, round(word_count / 100))

        return {
            "filename": file.filename,
            "extracted_text": extracted_text,
            "word_count": word_count,
            "estimated_minutes": est_minutes
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")