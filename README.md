# ReadMate Automated Reading Assessment API

Backend service for the **ReadMate** automated reading assessment and monitoring system. Built with FastAPI, OpenAI Whisper ASR, Levenshtein/WER alignment, a Scikit-Learn Random Forest classifier, and Firebase Firestore.

- **Live Base URL:** `[https://readmate-backend.onrender.com](https://readmate-backend.onrender.com)`
- **Interactive API Docs (Swagger UI):** `[https://readmate-backend.onrender.com/docs](https://readmate-backend.onrender.com/docs)`

---

## Core Features

1. **Speech-to-Text Transcription:** Uses OpenAI Whisper (`tiny` model, CPU-optimized) to transcribe student oral reading recordings in English (`en`) and Filipino (`tl`).
2. **Oral Reading Fluency & Accuracy Scoring:** Computes Word Error Rate (WER), reading accuracy percentage, Words Per Minute (WPM), fluency percentage, and word-level substitutions, deletions, and insertions.
3. **Reading Level Classification:** Uses a trained Random Forest model (`models/rf_reading_model.pkl`) to classify students into `Beginner`, `Developing`, `Proficient`, or `Advanced`.
4. **Zero-Retention Audio Privacy:** Uploaded audio files are processed in temporary storage and deleted immediately after transcription (`"audio_retained": false`).
5. **Passage Text Extraction:** Parses uploaded `.pdf`, `.docx`, and `.txt` reading materials and estimates reading duration.
6. **Cloud Database Sync:** Writes assessment records to the `assessments` collection and updates student progress in the `students` collection in Firebase Firestore.

---

## API Endpoints

### 1. Health Check
- **Endpoint:** `GET /`
- **Description:** Verifies that the API server is online. Frontend apps should ping this endpoint when opening the dashboard or reading screen to wake the Render free-tier container.

### 2. Assess Student Reading
- **Endpoint:** `POST /api/assess`
- **Content-Type:** `multipart/form-data`
- **Form Fields:**
  - `audio` (file): Audio recording (`.m4a`, `.wav`, or `.mp3`)
  - `student_id` (string): Unique student identifier
  - `student_name` (string): Full name of the student
  - `class_id` (string): Class or section identifier
  - `material_id` (string): Reading passage identifier
  - `material_title` (string): Title of the reading passage
  - `reference_text` (string): Original passage text
  - `language` (string): `"en"` for English or `"tl"` for Filipino
  - `duration_seconds` (number): Total reading duration in seconds
  - `student_answers_json` (string): JSON array of student quiz answers (e.g., `["Morning", "Over the quiet hills"]`)
  - `correct_answers_json` (string): JSON array of expected quiz answers (e.g., `["Morning", "Over the quiet hills"]`)

**Sample JSON Response (`200 OK`):**
```json
{
  "assessment_id": "4wltllfSqPxQd5bdacUk",
  "data": {
    "student_id": "student_01",
    "student_name": "Mia Fernandez",
    "class_id": "K3-MAS-2026",
    "material_id": "mat_01",
    "material_title": "The Morning Valley",
    "transcribed_text": "The sun was just beginning to rise over the white hills.",
    "total_words": 11,
    "words_read": 12,
    "duration_seconds": 5,
    "reading_accuracy": 81.82,
    "wer": 18.18,
    "wpm": 144,
    "fluency_percentage": 100,
    "comprehension_score": 100,
    "quiz_correct": 2,
    "quiz_total": 2,
    "overall_score": 90.91,
    "reading_level": "Advanced",
    "model_confidence": 53,
    "error_breakdown": {
      "substitutions": 1,
      "deletions": 0,
      "insertions": 1
    },
    "audio_retained": false
  }
}
```

### 3. Extract Reading Material Text
- **Endpoint:** `POST /api/materials/extract-text`
- **Content-Type:** `multipart/form-data`
- **Form Fields:**
  - `file` (file): Reading material document (`.pdf`, `.docx`, or `.txt`)

**Sample JSON Response (`200 OK`):**
```json
{
  "filename": "passage.pdf",
  "extracted_text": "The sun was just beginning to rise over the quiet hills...",
  "word_count": 120,
  "estimated_minutes": 1
}
```

---

## Frontend Integration Notes (Base44 & Flutter)

- **Cold Starts:** On Render's free tier, the server sleeps after 15 minutes of inactivity and takes 30 to 50 seconds to wake up. Send a background `GET [https://readmate-backend.onrender.com/](https://readmate-backend.onrender.com/)` request on app startup.
- **Request Timeout:** Set HTTP client timeouts to at least `60` seconds for `POST /api/assess`.

---

## Project Structure

```text
readmate-backend/
├── main.py                  # FastAPI routes, CORS, and Firestore integration
├── assessment_engine.py     # Whisper ASR, WER/Levenshtein scoring, and Random Forest inference
├── train_model.py           # Generates synthetic reading dataset and trains the Random Forest model
├── models/
│   └── rf_reading_model.pkl # Serialized Random Forest classifier
├── requirements.txt         # Python dependencies
└── .gitignore               # Excludes virtual environments, cache files, and Firebase credentials
```

---

## Local Setup

1. Create and activate a Python 3.11 virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies and train the classifier:
   ```bash
   pip install -r requirements.txt
   python train_model.py
   ```
3. Place your Firebase Admin SDK key as `serviceAccountKey.json` in the project root.
4. Start the local server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

---

## Render Deployment Configuration

- **Runtime:** Python 3 (`PYTHON_VERSION=3.11.9`)
- **Build Command:**
  ```bash
  pip install "setuptools<82" wheel && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-build-isolation openai-whisper==20240930 && pip install -r requirements.txt && python train_model.py
  ```
- **Start Command:**
  ```bash
  uvicorn main:app --host 0.0.0.0 --port $PORT
  ```
- **Environment Variable:** `FIREBASE_KEY_PATH=/etc/secrets/serviceAccountKey.json`
- **Secret File:** `serviceAccountKey.json`
